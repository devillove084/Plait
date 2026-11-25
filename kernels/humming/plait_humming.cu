#include "plait_humming.h"
#include "w4a16_config.cuh"

#include <cuda_runtime.h>
#include <humming/kernel/process.cuh>

namespace {

constexpr uint32_t n = LayerConfig::kShapeN;
constexpr uint32_t k = LayerConfig::kShapeK;
constexpr uint32_t group_size = LayerConfig::kWeightScaleGroupSize;
using BlockShape = Shape<HUMMING_BLOCK_SHAPE_M, HUMMING_BLOCK_SHAPE_N, HUMMING_BLOCK_SHAPE_K>;
using WarpShape = Shape<16, 64, 64>;
using F16 = FloatingPointType<16, 5, 10>;
using U4 = IntegerType<false, 4>;
using Storage = SharedStorage<MmaOpClass, BlockShape, WarpShape, F16, U4, F16,
                              LayerConfig, ComputeConfig, TuningConfig>;

static_assert(n > 0 && n % BlockShape::N == 0);
static_assert(k > 0 && k % group_size == 0);
static_assert(uint64_t(n) * k <= INT32_MAX);
static_assert(sizeof(Storage) <= 48 * 1024);

bool supports_shape(uint32_t rows, uint32_t columns, uint32_t group) {
    return rows == n && columns == k && group == group_size;
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
                        const size_t index = size_t(row) * n + col;
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

template <template <class, class, class> class OutputPipeline>
int launch(const plait_w4a16_shape &shape, const void *activation, const void *weight,
           const void *scale, const void *zero, void *output, void *stream) {
    const uint32_t m = shape.m;
    const uint32_t grid = (m / BlockShape::M + (m % BlockShape::M != 0)) * (n / BlockShape::N);
    humming<MmaOpClass, Shape<0, n, k>, BlockShape, WarpShape, Shape<0, 0, 0>,
            F16, U4, F16, F16, LayerConfig, ComputeConfig, TuningConfig, OutputPipeline>
        <<<grid, TuningConfig::kNumThreads, sizeof(Storage), static_cast<cudaStream_t>(stream)>>>(
            const_cast<void *>(activation), const_cast<void *>(weight), output,
            nullptr, nullptr, const_cast<void *>(scale), const_cast<void *>(zero),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, 1, false);
    return cudaGetLastError();
}

}  // namespace

int plait_humming_probe(int ordinal, uint32_t rows, uint32_t columns, uint32_t group) {
    if (!supports_shape(rows, columns, group)) return -1;
    cudaDeviceProp properties;
    const auto status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) return status;
    return properties.major * 10 + properties.minor == PLAIT_HUMMING_SM ? 0 : -1;
}

int plait_humming_repack(const void *canonical, void *packed, void *stream) {
    const dim3 grid(n / 64, k / 64, 1);
    weight_repack_nk<4, 16, true, false, false, false, 0>
        <<<grid, 32, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint32_t *>(canonical), static_cast<uint32_t *>(packed),
            nullptr, n, k, n, k, 3);
    return cudaGetLastError();
}

int plait_humming_launch(const plait_w4a16_shape *shape, const void *activation,
                         const void *weight, const void *scale, const void *zero,
                         void *output, void *stream) {
    if (!shape || !shape->m || !supports_shape(shape->n, shape->k, shape->group_size)) {
        return cudaErrorInvalidValue;
    }
    if (shape->output_bits == 16) {
        return launch<EpiloguePipeline>(*shape, activation, weight, scale, zero, output, stream);
    }
#if defined(PLAIT_W4_HUMMING_SHARDS)
    if (shape->output_bits == 32) {
        return launch<F32Output>(*shape, activation, weight, scale, zero, output, stream);
    }
#endif
    return cudaErrorInvalidValue;
}
