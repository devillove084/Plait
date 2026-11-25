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
    if (!supports_shape(shape->n, shape->k, shape->group_size) || shape->output_bits != 16) {
        return cudaErrorInvalidValue;
    }
    const uint32_t m = shape->m;
    const uint32_t grid = (m / BlockShape::M + (m % BlockShape::M != 0)) * (n / BlockShape::N);
    humming<MmaOpClass, Shape<0, n, k>, BlockShape, WarpShape, Shape<0, 0, 0>,
            F16, U4, F16, F16, LayerConfig, ComputeConfig, TuningConfig>
        <<<grid, TuningConfig::kNumThreads, sizeof(Storage), static_cast<cudaStream_t>(stream)>>>(
            const_cast<void *>(activation), const_cast<void *>(weight), output,
            nullptr, nullptr, const_cast<void *>(scale), const_cast<void *>(zero),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, 1, false);
    return cudaGetLastError();
}
