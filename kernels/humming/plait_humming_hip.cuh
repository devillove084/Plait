#pragma once

// Dense Humming W4A16 tiling and staged loads on CDNA3. The physical
// weight layout follows MFMA lane ownership and is private to this backend.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "The Humming MFMA backend is compiled for gfx942"
#endif

namespace plait_humming_mfma {

constexpr uint32_t wave = 64;
constexpr uint32_t threads = 4 * wave;
constexpr uint32_t mma = 16;
constexpr uint32_t tile_m = 32;
constexpr uint32_t tile_n = 64;
constexpr uint32_t tile_k = 64;
constexpr uint32_t group_k = 128;
constexpr uint32_t words_per_stage = tile_n * tile_k / 4;
constexpr uint32_t a_per_row = tile_k / 4;

using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Codes4 = uint16_t __attribute__((ext_vector_type(4)));

inline bool supports(uint32_t n, uint32_t k, uint32_t group) {
    return n && n % tile_n == 0 && k && k % group_k == 0 &&
           group == group_k && uint64_t(n) * k <= INT32_MAX;
}

// A stage holds four MFMA steps. Each step has four adjacent column
// fragments with one word of four INT4 values per wave lane.
__global__ void repack(const uint16_t *canonical, uint16_t *packed, uint32_t k,
                       uint32_t word_count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= word_count) return;
    const uint32_t lane = index % wave;
    const uint32_t fragment = index / wave % (tile_n / mma);
    const uint32_t step = index / (wave * (tile_n / mma)) % (tile_k / mma);
    const uint32_t stages_k = k / tile_k;
    const uint32_t stage = index / words_per_stage;
    const uint32_t column = stage / stages_k * tile_n + fragment * mma + lane % mma;
    const uint32_t inner = stage % stages_k * tile_k + step * mma + lane / mma * 4;
    packed[index] = canonical[(size_t(column) * k + inner) / 4];
}

struct Stage {
    Half4 activation[tile_m * a_per_row];
    Codes4 weight[words_per_stage / 4];
};
static_assert(words_per_stage == threads * 4);

__device__ inline void load_stage(Stage &stage, const Half4 *activation, const Codes4 *weight,
                                  plait_w4a16_shape shape, uint32_t first_row, uint32_t inner) {
    const uint32_t lane = threadIdx.x;
    #pragma unroll
    for (uint32_t index = lane; index < tile_m * a_per_row; index += threads) {
        const uint32_t row = first_row + index / a_per_row;
        stage.activation[index] = row < shape.m
            ? activation[(size_t(row) * shape.k + inner) / 4 + index % a_per_row]
            : Half4{};
    }
    stage.weight[lane] = weight[lane];
}

struct Parameters {
    _Float16 scale;
    _Float16 zero;
};

__device__ inline _Float16 half_bits(uint16_t value) {
    return __builtin_bit_cast(_Float16, value);
}

__device__ inline Parameters parameters(const uint16_t *scales, const uint8_t *zeros,
                                        uint32_t column) {
    // Humming's shared group layout pairs channels c and c+8 in each
    // 16-channel fragment and orders pairs by lane within a 64-channel tile.
    const uint32_t pair = column % 8 * 4 + column / mma;
    const uint32_t half = column / 8 % 2;
    const uint32_t zero = (zeros[pair] >> (half * 4)) & 15;
    return {half_bits(scales[2 * pair + half]), half_bits(uint16_t(0x6400u + zero))};
}

__device__ inline Half4 decode(uint32_t codes, Parameters param) {
    // FP16 has unit spacing at 1024. All nibble values and their zero
    // point can share this exponent until the exact integer subtraction.
    const Half4 values{
        half_bits(uint16_t(0x6400u | (codes & 15))),
        half_bits(uint16_t(0x6400u | ((codes >> 4) & 15))),
        half_bits(uint16_t(0x6400u | ((codes >> 8) & 15))),
        half_bits(uint16_t(0x6400u | ((codes >> 12) & 15))),
    };
    return (values - param.zero) * param.scale;
}

template<bool F32>
__device__ inline void store_rows(void *output, plait_w4a16_shape shape, uint32_t row,
                                  uint32_t column, Float4 accum) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        if (row + i >= shape.m) break;
        const size_t index = size_t(row + i) * shape.n + column;
        if constexpr (F32) static_cast<float *>(output)[index] = accum[i];
        else static_cast<_Float16 *>(output)[index] = _Float16(accum[i]);
    }
}

// A single row tile has little activation reuse between waves. Assign each
// block one column fragment and split long K over its waves. All scratch is
// local to that block and the epilogue adds the waves in K order.
template<bool F32, uint32_t Parts>
__global__ void project_narrow(const Half4 *activation, const uint16_t *weight,
                                const uint16_t *scales, const uint8_t *zeros,
                                void *output, plait_w4a16_shape shape) {
    const uint32_t lane = threadIdx.x % wave;
    const uint32_t part = threadIdx.x / wave;
    const uint32_t column = blockIdx.x * mma + lane % mma;
    const uint32_t column_tile = column / tile_n;
    const uint32_t fragment = column % tile_n / mma;
    const uint32_t a_row = lane % mma;
    const uint32_t groups = shape.k / group_k;
    const uint32_t first = part * groups / Parts;
    const uint32_t end = (part + 1) * groups / Parts;
    const uint16_t *tile_weight = weight + size_t(column_tile) * (shape.k / tile_k) * words_per_stage;
    Float4 accum{};

    for (uint32_t group = first; group < end; ++group) {
        const size_t scale_offset = size_t(group) * shape.n + column_tile * tile_n;
        const auto param = parameters(scales + scale_offset, zeros + scale_offset / 2, column % tile_n);
        #pragma unroll
        for (uint32_t step = 0; step < group_k / mma; ++step) {
            Half4 a{};
            if (a_row < shape.m) {
                a = activation[(size_t(a_row) * shape.k + group * group_k + step * mma) / 4 + lane / mma];
            }
            const uint32_t k_tile = group * (group_k / tile_k) + step / (tile_k / mma);
            const uint32_t word = k_tile * words_per_stage +
                (step % (tile_k / mma) * (tile_n / mma) + fragment) * wave + lane;
            accum = __builtin_amdgcn_mfma_f32_16x16x16f16(a, decode(tile_weight[word], param),
                                                            accum, 0, 0, 0);
        }
    }

    if constexpr (Parts > 1) {
        __shared__ Float4 partial[Parts][wave];
        partial[part][lane] = accum;
        __syncthreads();
        if (part != 0) return;
        #pragma unroll
        for (uint32_t next = 1; next < Parts; ++next) accum += partial[next][lane];
    }
    store_rows<F32>(output, shape, lane / mma * 4, column, accum);
}

template<bool F32>
__global__ void project(const Half4 *activation, const Codes4 *weight, const uint16_t *scales,
                         const uint8_t *zeros, void *output, plait_w4a16_shape shape) {
    __shared__ Stage pipeline[2];
    const uint32_t lane = threadIdx.x % wave;
    const uint32_t wave_id = threadIdx.x / wave;
    const uint32_t row_fragment = wave_id / 2 * mma;
    const uint32_t column_fragment = wave_id % 2 * 2;
    const uint32_t column_tiles = shape.n / tile_n;
    const uint32_t column_tile = blockIdx.x % column_tiles;
    const uint32_t first_row = blockIdx.x / column_tiles * tile_m;
    const uint32_t first_column = column_tile * tile_n;
    const uint32_t stages = shape.k / tile_k;
    constexpr uint32_t stage_vectors = words_per_stage / 4;
    const Codes4 *tile_weight = weight + size_t(column_tile) * stages * stage_vectors;
    const uint32_t a_row = (row_fragment + lane % mma) * a_per_row;
    Parameters param[2];
    Float4 accum[2]{};

    load_stage(pipeline[0], activation, tile_weight, shape, first_row, 0);
    __syncthreads();
    for (uint32_t stage = 0; stage < stages; ++stage) {
        if (stage + 1 < stages) {
            load_stage(pipeline[(stage + 1) % 2], activation,
                       tile_weight + (stage + 1) * stage_vectors, shape,
                       first_row, (stage + 1) * tile_k);
        }
        if (stage % (group_k / tile_k) == 0) {
            const size_t group = size_t(stage / (group_k / tile_k)) * shape.n + first_column;
            #pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                const uint32_t column = (column_fragment + j) * mma + lane % mma;
                param[j] = parameters(scales + group, zeros + group / 2, column);
            }
        }
        const Stage &current = pipeline[stage % 2];
        #pragma unroll
        for (uint32_t step = 0; step < tile_k / mma; ++step) {
            const Half4 a = current.activation[a_row + step * (mma / 4) + lane / mma];
            #pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                const uint32_t fragment = step * (tile_n / mma) + column_fragment + j;
                const uint32_t word = fragment * wave + lane;
                const Half4 b = decode(current.weight[word / 4][word % 4], param[j]);
                accum[j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, accum[j], 0, 0, 0);
            }
        }
        // Publish the prefetched tile and retire every reader of the
        // current tile before another wave can reuse its buffer.
        __syncthreads();
    }

    const uint32_t row = first_row + row_fragment + lane / mma * 4;
    const uint32_t column = first_column + column_fragment * mma + lane % mma;
    #pragma unroll
    for (uint32_t j = 0; j < 2; ++j) store_rows<F32>(output, shape, row, column + j * mma, accum[j]);
}

template<bool F32>
inline hipError_t launch(hipStream_t stream, const plait_w4a16_shape &shape,
                         const Half4 *activation, const void *weight, const uint16_t *scales,
                         const uint8_t *zeros, void *output) {
    if (shape.m <= mma) {
        const uint32_t blocks = shape.n / mma;
        const auto *weights = static_cast<const uint16_t *>(weight);
        if (shape.k < 4 * group_k) {
            project_narrow<F32, 1><<<blocks, wave, 0, stream>>>(activation, weights, scales, zeros, output, shape);
        } else {
            project_narrow<F32, 4><<<blocks, 4 * wave, 0, stream>>>(activation, weights, scales, zeros, output, shape);
        }
    } else {
        const uint32_t blocks = shape.n / tile_n * plait_ceil_div(shape.m, tile_m);
        project<F32><<<blocks, threads, 0, stream>>>(
            activation, static_cast<const Codes4 *>(weight), scales, zeros, output, shape);
    }
    return hipGetLastError();
}

}  // namespace plait_humming_mfma

inline int plait_humming_probe_shape(int ordinal, uint32_t n, uint32_t k, uint32_t group,
                                    uint32_t output_bits) {
    if (!plait_humming_mfma::supports(n, k, group) ||
        (output_bits != 16 && output_bits != 32)) return -1;
    return probe_gfx942(ordinal);
}

inline int plait_humming_repack_shape(uint32_t n, uint32_t k, uint32_t group,
                                      const void *canonical, void *packed, void *stream) {
    if (!plait_humming_mfma::supports(n, k, group) || !canonical || !packed) {
        return hipErrorInvalidValue;
    }
    using namespace plait_humming_mfma;
    const uint32_t words = n * k / 4;
    repack<<<plait_ceil_div(words, threads), threads, 0, static_cast<hipStream_t>(stream)>>>(
        static_cast<const uint16_t *>(canonical), static_cast<uint16_t *>(packed), k, words);
    return hipGetLastError();
}

inline int plait_humming_launch_shape(const plait_w4a16_shape *shape, const void *activation,
                                       const void *weight, const void *scale, const void *zero,
                                       void *output, void *stream) {
    using namespace plait_humming_mfma;
    if (!plait_w4a16_valid_shape(shape) || !supports(shape->n, shape->k, shape->group_size) ||
        uint64_t(shape->m) * shape->n > INT32_MAX ||
        uint64_t(shape->m) * shape->k > INT32_MAX) {
        return hipErrorInvalidValue;
    }
    const auto *a = static_cast<const Half4 *>(activation);
    const auto *s = static_cast<const uint16_t *>(scale);
    const auto *z = static_cast<const uint8_t *>(zero);
    const auto queue = static_cast<hipStream_t>(stream);
    return shape->output_bits == 32 ? launch<true>(queue, *shape, a, weight, s, z, output)
                                   : launch<false>(queue, *shape, a, weight, s, z, output);
}
