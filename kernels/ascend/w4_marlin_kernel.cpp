#include "w4_device_util.h"

using plait_ascend::WriteSlices;
using plait_ascend::flush;

namespace {

constexpr uint32_t k_unit = PLAIT_ASCEND_MARLIN_K_UNIT;
constexpr uint32_t tile_m = PLAIT_ASCEND_TILE_M;
constexpr uint32_t tile_n = PLAIT_ASCEND_TILE_N;

__aicore__ void pack_weights(GM_ADDR weight, GM_ADDR params, GM_ADDR packed, uint32_t cols,
    uint32_t n, uint32_t k, uint32_t group_size, uint32_t col_start, uint32_t padded_cols,
    uint32_t depth, uint32_t first, uint32_t groups) {
    const auto source = reinterpret_cast<__gm__ const uint16_t *>(weight);
    const auto parameters = reinterpret_cast<__gm__ const uint32_t *>(params);
    const auto out = reinterpret_cast<__gm__ half *>(packed);
    const uint64_t group_stride = uint64_t(padded_cols) * depth;
    const WriteSlices slices(packed, group_stride * groups, sizeof(half));
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t pos = slices.start(slice); pos < end; pos += 4) {
            const uint32_t group = first + pos / group_stride;
            const uint32_t col = (pos % group_stride) / depth;
            const uint32_t inner = group * k_unit + pos % depth;
            int zero = 0;
            uint16_t codes = 0;
            if (col < cols && inner < k) {
                const uint32_t channel = col_start + col;
                const uint64_t tile = uint64_t(channel / 16) * (k / 16) + inner / 16;
                codes = source[tile * 64 + (inner % 16) / 4 * 16 + channel % 16];
                const uint32_t scale_group = group_size == k ? 0 : group;
                zero = parameters[uint64_t(scale_group) * n + channel] >> 16;
            }
            // Centered INT4 codes are exactly representable in binary16.
            for (uint32_t i = 0; i < 4; ++i) {
                out[pos + i] = half(int((codes >> (4 * i)) & 15) - zero);
            }
        }
    }
    flush(packed);
}

}  // namespace

extern "C" __global__ __aicore__ void plait_ascend_marlin_pack(
    GM_ADDR activation, GM_ADDR weight, GM_ADDR params, GM_ADDR packed_a, GM_ADDR packed_b,
    uint32_t rows, uint32_t cols, uint32_t n, uint32_t k, uint32_t group_size, uint32_t col_start,
    uint32_t padded_rows, uint32_t padded_cols, uint32_t depth, uint32_t first, uint32_t groups) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    plait_ascend::pack_projection_activation(
        activation, packed_a, rows, k, padded_rows, depth, k_unit, first, groups);
    pack_weights(weight, params, packed_b, cols, n, k, group_size, col_start,
                 padded_cols, depth, first, groups);
}

extern "C" __global__ __aicore__ void plait_ascend_marlin_finish(
    GM_ADDR dots, GM_ADDR params, GM_ADDR sums, GM_ADDR output,
    uint32_t rows, uint32_t cols, uint32_t n, uint32_t col_start, uint32_t padded_rows,
    uint32_t padded_cols, uint32_t first, uint32_t groups, uint32_t last,
    uint32_t channelwise, uint32_t output_bits) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto partials = reinterpret_cast<__gm__ const float *>(dots);
    const auto scales = reinterpret_cast<__gm__ const half *>(params);
    const auto running = reinterpret_cast<__gm__ float *>(sums);
    const uint32_t column_tiles = padded_cols / tile_n;
    const uint32_t tiles = padded_rows / tile_m * column_tiles;
    const uint32_t stride = last ? n : padded_cols;
    GM_ADDR destination = last ? output : sums;
    const WriteSlices slices(destination, uint64_t(rows) * stride, last ? output_bits / 8 : 4);
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t index = slices.start(slice); index < end; ++index) {
            const uint32_t row = index / stride;
            const uint32_t physical_col = index % stride;
            if (last && physical_col < col_start) continue;
            const uint32_t col = physical_col - (last ? col_start : 0);
            if (col >= cols) continue;
            const uint32_t channel = col_start + col;
            const uint64_t sum_index = uint64_t(row) * padded_cols + col;
            const uint64_t tile = uint64_t(row / tile_m) * column_tiles + col / tile_n;
            const uint32_t in_tile = (row % tile_m) * tile_n + col % tile_n;
            float sum = first ? running[sum_index] : 0.0f;
            for (uint32_t unit = 0; unit < groups; ++unit) {
                const float dot = partials[(uint64_t(unit) * tiles + tile) * (tile_m * tile_n) + in_tile];
                const float scale = channelwise ? 1.0f : float(scales[2 * (uint64_t(first + unit) * n + channel)]);
                sum += dot * scale;
            }
            if (!last) {
                running[sum_index] = sum;
                continue;
            }
            if (channelwise) sum *= float(scales[2 * uint64_t(channel)]);
            if (output_bits == 32) reinterpret_cast<__gm__ float *>(output)[index] = sum;
            else reinterpret_cast<__gm__ half *>(output)[index] = half(sum);
        }
    }
    flush(destination);
}
