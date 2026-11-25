#include "plait_humming.h"
#include "w4a16_config.cuh"

#include <cuda_runtime.h>
#include <humming/kernel/process.cuh>

namespace {

constexpr uint32_t primary_n = PLAIT_HUMMING_N;
constexpr uint32_t primary_k = PLAIT_HUMMING_K;
constexpr uint32_t group_size = LayerConfig<primary_n, primary_k>::kWeightScaleGroupSize;
using BlockShape = Shape<HUMMING_BLOCK_SHAPE_M, HUMMING_BLOCK_SHAPE_N, HUMMING_BLOCK_SHAPE_K>;
using WarpShape = Shape<16, 64, 64>;
using F16 = FloatingPointType<16, 5, 10>;
using U4 = IntegerType<false, 4>;
bool primary_shape(uint32_t n, uint32_t k, uint32_t group) {
    return n == primary_n && k == primary_k && group == group_size;
}

#if defined(PLAIT_W4_HUMMING_SHARDS)
// This dense specialization has one K warp and no output scales, bias, or
// split K. Store the accumulator pairs directly as row-major FP32.
template <class Ctx, class MMA, class Arithmetic>
class F32Output : public EpiloguePipeline<Ctx, MMA, Arithmetic> {
public:
    static_assert(Ctx::kIsDenseGemm && !Ctx::TuningConfig::kUseStreamK);
    static_assert(!Ctx::kUseWgmma && !Ctx::kUseUmma && !Ctx::kHasInputScale);
    static_assert(!Ctx::kHasTensorWeightScale && !Arithmetic::kNeedsPackedOutputTransform);
    static_assert(BlockShape::K == WarpShape::K);
    using Base = EpiloguePipeline<Ctx, MMA, Arithmetic>;
    using Base::Base;

    __device__ void seek(uint32_t, uint32_t m_block, uint32_t n_block, uint32_t rows, uint32_t) {
        first_row = m_block * BlockShape::M;
        first_column = n_block * BlockShape::N;
        output_rows = rows;
    }

    __device__ void call(uint32_t *registers) {
        using Registers = typename MMA::CRegistersArrayType;
        const auto &tiles = *reinterpret_cast<const Registers *>(registers);
        const uint32_t lane = this->ctx.lane_id();
        const uint32_t row_base = first_row + this->ctx.m_warp_offset() + lane / 4;
        const uint32_t col_base = first_column + this->ctx.n_warp_id() * WarpShape::N + (lane % 4) * 2;
        auto *output = static_cast<float *>(const_cast<void *>(this->ctx.params.c));
        static_assert(sizeof(tiles[0][0]) == 2 * sizeof(float2));
        #pragma unroll
        for (uint32_t i = 0; i < sizeof(tiles) / sizeof(tiles[0]); ++i) {
            #pragma unroll
            for (uint32_t j = 0; j < sizeof(tiles[0]) / sizeof(tiles[0][0]); ++j) {
                const auto *pair = reinterpret_cast<const float2 *>(&tiles[i][j]);
                #pragma unroll
                for (uint32_t part = 0; part < 2; ++part) {
                    const uint32_t row = row_base + 8 * (i * 2 + (USE_PPU ? 0 : part));
                    const uint32_t col = col_base + 8 * (j + (USE_PPU ? part : 0));
                    if (row < output_rows) {
                        const size_t index = size_t(row) * Ctx::LayerConfig::kShapeN + col;
                        output[index] = pair[part].x;
                        output[index + 1] = pair[part].y;
                    }
                }
            }
        }
    }

private:
    uint32_t first_row, first_column, output_rows;
};
#endif

template <uint32_t N, uint32_t K, template <class, class, class> class OutputPipeline>
int launch(const plait_w4a16_shape &shape, const void *activation, const void *weight,
           const void *scale, const void *zero, void *output, void *stream) {
    using Layer = LayerConfig<N, K>;
    using Storage = SharedStorage<MmaOpClass, BlockShape, WarpShape, F16, U4, F16,
                                  Layer, ComputeConfig, TuningConfig>;
    static_assert(N > 0 && N % BlockShape::N == 0);
    static_assert(K > 0 && K % group_size == 0);
    static_assert(uint64_t(N) * K <= INT32_MAX);
    static_assert(sizeof(Storage) <= 48 * 1024);
    const uint32_t m = shape.m;
    if (uint64_t(m) * N > INT32_MAX || uint64_t(m) * K > INT32_MAX) return cudaErrorInvalidValue;
    const uint32_t grid = (m / BlockShape::M + (m % BlockShape::M != 0)) * (N / BlockShape::N);
    humming<MmaOpClass, Shape<0, N, K>, BlockShape, WarpShape, Shape<0, 0, 0>,
            F16, U4, F16, F16, Layer, ComputeConfig, TuningConfig, OutputPipeline>
        <<<grid, TuningConfig::kNumThreads, sizeof(Storage), static_cast<cudaStream_t>(stream)>>>(
            const_cast<void *>(activation), const_cast<void *>(weight), output,
            nullptr, nullptr, const_cast<void *>(scale), const_cast<void *>(zero),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, 1, false);
    return cudaGetLastError();
}

template <uint32_t N, uint32_t K>
int launch_shape(const plait_w4a16_shape &shape, const void *activation, const void *weight,
                 const void *scale, const void *zero, void *output, void *stream) {
    if (shape.output_bits == 16) {
        return launch<N, K, EpiloguePipeline>(shape, activation, weight, scale, zero, output, stream);
    }
#if defined(PLAIT_W4_HUMMING_SHARDS)
    if (shape.output_bits == 32) {
        return launch<N, K, F32Output>(shape, activation, weight, scale, zero, output, stream);
    }
#endif
    return cudaErrorInvalidValue;
}

struct ShapeEntry {
    uint32_t n, k;
    decltype(&launch_shape<primary_n, primary_k>) launch;
};

constexpr ShapeEntry catalog[] = {
    {primary_n, primary_k, launch_shape<primary_n, primary_k>},
#if defined(PLAIT_HUMMING_SHAPES_FILE)
#define PLAIT_HUMMING_SHAPE(N, K) {N, K, launch_shape<N, K>},
#include PLAIT_HUMMING_SHAPES_FILE
#undef PLAIT_HUMMING_SHAPE
#endif
};

constexpr bool unique_shapes() {
    for (size_t i = 0; i < sizeof(catalog) / sizeof(catalog[0]); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (catalog[i].n == catalog[j].n && catalog[i].k == catalog[j].k) return false;
        }
    }
    return true;
}
static_assert(unique_shapes(), "Humming catalog contains a duplicate shape");

const ShapeEntry *find_shape(uint32_t n, uint32_t k, uint32_t group) {
    if (group != group_size) return nullptr;
    for (const auto &entry : catalog) {
        if (entry.n == n && entry.k == k) return &entry;
    }
    return nullptr;
}

int probe_device(int ordinal) {
    cudaDeviceProp properties;
    const auto status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) return status;
    return properties.major * 10 + properties.minor == PLAIT_HUMMING_SM ? 0 : -1;
}

int repack(uint32_t n, uint32_t k, const void *canonical, void *packed, void *stream) {
    const dim3 grid(n / 64, k / 64, 1);
    weight_repack_nk<4, 16, true, false, false, false, 0>
        <<<grid, 32, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint32_t *>(canonical), static_cast<uint32_t *>(packed),
            nullptr, n, k, n, k, 3);
    return cudaGetLastError();
}

}  // namespace

int plait_humming_probe(int ordinal, uint32_t n, uint32_t k, uint32_t group) {
    return primary_shape(n, k, group) ? probe_device(ordinal) : -1;
}

int plait_humming_probe_shape(int ordinal, uint32_t n, uint32_t k, uint32_t group, uint32_t bits) {
#if defined(PLAIT_W4_HUMMING_SHARDS)
    if (bits != 16 && bits != 32) return -1;
#else
    if (bits != 16) return -1;
#endif
    return find_shape(n, k, group) ? probe_device(ordinal) : -1;
}

int plait_humming_repack(const void *canonical, void *packed, void *stream) {
    return repack(primary_n, primary_k, canonical, packed, stream);
}

int plait_humming_repack_shape(uint32_t n, uint32_t k, uint32_t group, const void *canonical,
                               void *packed, void *stream) {
    if (!canonical || !packed || !find_shape(n, k, group)) return cudaErrorInvalidValue;
    return repack(n, k, canonical, packed, stream);
}

int plait_humming_launch(const plait_w4a16_shape *shape, const void *activation,
                         const void *weight, const void *scale, const void *zero,
                         void *output, void *stream) {
    if (!shape || !shape->m || !primary_shape(shape->n, shape->k, shape->group_size)) {
        return cudaErrorInvalidValue;
    }
    return catalog[0].launch(*shape, activation, weight, scale, zero, output, stream);
}

int plait_humming_launch_shape(const plait_w4a16_shape *shape, const void *activation,
                               const void *weight, const void *scale, const void *zero,
                               void *output, void *stream) {
    if (!shape || !shape->m) return cudaErrorInvalidValue;
    const auto *entry = find_shape(shape->n, shape->k, shape->group_size);
    return entry ? entry->launch(*shape, activation, weight, scale, zero, output, stream)
                 : cudaErrorInvalidValue;
}
