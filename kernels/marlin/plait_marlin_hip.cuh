#pragma once

// CDNA3 weight-only INT4 GEMM using native FP16 MFMA and FP32 accumulation.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "The Marlin MFMA backend is compiled for gfx942"
#endif

namespace plait_marlin_mfma {

constexpr uint32_t tile = 16;
constexpr uint32_t wave = 64;
constexpr uint32_t waves_per_block = 4;
constexpr uint32_t k_unit = 128;
constexpr uint32_t max_parts = 16;
constexpr uint32_t target_blocks = 256;

struct Grid {
    uint32_t column_blocks;
    uint32_t tiles;
    uint32_t parts;
};

inline Grid grid(uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t column_blocks = plait_ceil_div(n, tile * waves_per_block);
    const uint32_t tiles = plait_ceil_div(m, tile) * column_blocks;
    if (!tiles || !k) return {column_blocks, tiles, 1};
    const uint32_t units = plait_ceil_div(k, k_unit);
    const uint32_t desired = plait_ceil_div(target_blocks, tiles);
    const uint32_t parts = desired < units ? desired : units;
    return {column_blocks, tiles, parts < max_parts ? parts : max_parts};
}

inline size_t workspace(uint32_t m, uint32_t n, uint32_t k) {
    const uint64_t elements = uint64_t(m) * n;
    if (elements > UINT32_MAX) return SIZE_MAX;
    const uint32_t parts = grid(m, n, k).parts;
    const size_t stride = parts * sizeof(float);
    if (elements > SIZE_MAX / stride) return SIZE_MAX;
    return parts == 1 ? 0 : size_t(elements) * stride;
}

using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

__device__ inline _Float16 from_bits(uint16_t bits) {
    return __builtin_bit_cast(_Float16, bits);
}

__device__ inline float scale(uint32_t parameter) {
    return float(from_bits(uint16_t(parameter)));
}

__device__ inline void store(void *output, size_t index, float value, uint32_t output_bits) {
    if (output_bits == 32) static_cast<float *>(output)[index] = value;
    else static_cast<_Float16 *>(output)[index] = _Float16(value);
}

template<bool Grouped>
__global__ void project(const Half4 *activation, const uint16_t *packed,
                        const uint32_t *params, float *scratch, void *output,
                        plait_w4a16_shape shape, Grid plan) {
    const uint32_t part = blockIdx.x / plan.tiles;
    const uint32_t block_tile = blockIdx.x % plan.tiles;
    const uint32_t row_start = block_tile / plan.column_blocks * tile;
    const uint32_t column_tile = block_tile % plan.column_blocks * waves_per_block + threadIdx.x / wave;
    if (column_tile >= shape.n / tile) return;

    const uint32_t lane = threadIdx.x % wave;
    const uint32_t row_input = row_start + lane % tile;
    const uint32_t column = column_tile * tile + lane % tile;
    const uint32_t units = (shape.k + k_unit - 1) / k_unit;
    const uint32_t first = part * units / plan.parts;
    const uint32_t end = (part + 1) * units / plan.parts;
    const uint16_t *weight_tile = packed + size_t(column_tile) * (shape.k / tile) * wave;
    Float4 accum{};

    for (uint32_t unit = first; unit < end; ++unit) {
        const uint32_t parameter = params[(Grouped ? size_t(unit) * shape.n : 0) + column];
        const _Float16 zero = from_bits(uint16_t(0x6400u + (parameter >> 16)));
        const _Float16 weight_scale = from_bits(uint16_t(parameter));
        const uint32_t stop = min(shape.k, (unit + 1) * k_unit);
        for (uint32_t inner = unit * k_unit; inner < stop; inner += tile) {
            // The four K lanes each carry four neighboring FP16 values.
            Half4 a{};
            if (row_input < shape.m) {
                a = activation[(size_t(row_input) * shape.k + inner) / 4 + lane / tile];
            }
            const uint32_t codes = weight_tile[(inner / tile) * wave + lane];
            // 0x6400 is FP16 1024, with unit spacing over all sixteen code values.
            // Subtract the zero point at the same exponent to decode exact integers.
            Half4 b{
                from_bits(uint16_t(0x6400u | (codes & 15))),
                from_bits(uint16_t(0x6400u | ((codes >> 4) & 15))),
                from_bits(uint16_t(0x6400u | ((codes >> 8) & 15))),
                from_bits(uint16_t(0x6400u | (codes >> 12))),
            };
            b -= zero;
            if constexpr (Grouped) b *= weight_scale;
            accum = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, accum, 0, 0, 0);
        }
    }

    const uint32_t row_output = row_start + (lane / tile) * 4;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        if (row_output + i >= shape.m) break;
        const size_t index = size_t(row_output + i) * shape.n + column;
        if (plan.parts > 1) {
            scratch[size_t(part) * shape.m * shape.n + index] = accum[i];
        } else {
            const float sum = Grouped ? accum[i] : accum[i] * scale(params[column]);
            store(output, index, sum, shape.output_bits);
        }
    }
}

__global__ void combine(const float *scratch, const uint32_t *params, void *output,
                        plait_w4a16_shape shape, uint32_t parts) {
    const size_t size = size_t(shape.m) * shape.n;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float sum = 0.0f;
    for (uint32_t part = 0; part < parts; ++part) sum += scratch[part * size + index];
    if (shape.k == shape.group_size) sum *= scale(params[index % shape.n]);
    store(output, index, sum, shape.output_bits);
}

inline hipError_t launch(hipStream_t stream, const plait_w4a16_shape &shape,
                         const void *activation, const void *weight, const void *scale_and_zero,
                         void *scratch, void *output) {
    const auto plan = grid(shape.m, shape.n, shape.k);
    if (!plan.tiles || (plan.parts > 1 && !scratch)) return hipErrorInvalidValue;
    constexpr uint32_t threads = wave * waves_per_block;
    const auto *a = static_cast<const Half4 *>(activation);
    const auto *w = static_cast<const uint16_t *>(weight);
    const auto *params = static_cast<const uint32_t *>(scale_and_zero);
    auto *partial = static_cast<float *>(scratch);
    const auto blocks = plan.tiles * plan.parts;
    if (shape.group_size == shape.k) {
        project<false><<<blocks, threads, 0, stream>>>(a, w, params, partial, output, shape, plan);
    } else {
        project<true><<<blocks, threads, 0, stream>>>(a, w, params, partial, output, shape, plan);
    }
    auto status = hipGetLastError();
    if (status == hipSuccess && plan.parts > 1) {
        combine<<<plait_ceil_div(shape.m * shape.n, threads), threads, 0, stream>>>(
            partial, params, output, shape, plan.parts);
        status = hipGetLastError();
    }
    return status;
}

}  // namespace plait_marlin_mfma
