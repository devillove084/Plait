#include "w4_device_util.h"
#include "w4_humming_layout.h"

using plait_ascend::WriteSlices;
using plait_ascend::flush;

namespace {

constexpr uint32_t tile_n = PLAIT_ASCEND_HUMMING_N_TILE;
constexpr uint32_t group_k = PLAIT_ASCEND_HUMMING_GROUP;
constexpr uint32_t words = group_k / 4;

__aicore__ uint64_t weight_group(uint32_t channel, uint32_t group, uint32_t groups) {
    return ((uint64_t(channel / tile_n) * groups + group) * tile_n + channel % tile_n) * words;
}

__aicore__ void pack_weights(GM_ADDR weight, GM_ADDR scale, GM_ADDR zero, GM_ADDR packed,
    uint32_t n, uint32_t k, uint32_t col_start, uint32_t cols, uint32_t depth,
    uint32_t first, uint32_t parts) {
    const auto source = reinterpret_cast<__gm__ const uint16_t *>(weight);
    const auto scales = reinterpret_cast<__gm__ const half *>(scale);
    const auto zeros = reinterpret_cast<__gm__ const uint8_t *>(zero);
    const auto output = reinterpret_cast<__gm__ half *>(packed);
    const uint32_t groups = k / group_k;
    const uint64_t part_stride = uint64_t(cols) * depth;
    // Each workspace panel starts on a 1 KiB boundary. A writer owns full
    // cache lines and decodes one complete scale group at a time.
    const WriteSlices slices(packed, part_stride * parts, sizeof(half));
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t pos = slices.start(slice); pos < end; pos += group_k) {
            const uint32_t part = first + pos / part_stride;
            const uint32_t channel = col_start + (pos % part_stride) / depth;
            const uint32_t group = (part * depth + pos % depth) / group_k;
            if (group >= groups) {
                for (uint32_t i = 0; i < group_k; ++i) output[pos + i] = half(0);
                continue;
            }
            const uint32_t local = channel % tile_n;
            const uint32_t pair = (local % 8) * 4 + local / 16;
            const uint32_t high = local / 8 % 2;
            const uint64_t offset = uint64_t(group) * n + channel - local;
            const float multiplier = float(scales[offset + 2 * pair + high]);
            const int z = (zeros[offset / 2 + pair] >> (4 * high)) & 15;
            const auto packed_group = source + weight_group(channel, group, groups);
            for (uint32_t word = 0; word < words; ++word) {
                const uint16_t codes = packed_group[word];
                for (uint32_t i = 0; i < 4; ++i) {
                    const int centered = int((codes >> (4 * i)) & 15) - z;
                    output[pos + 4 * word + i] = half(float(centered) * multiplier);
                }
            }
        }
    }
    flush(packed);
}

}  // namespace

extern "C" __global__ __aicore__ void plait_ascend_humming_repack(
    GM_ADDR canonical, GM_ADDR packed, uint32_t n, uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto source = reinterpret_cast<__gm__ const uint64_t *>(canonical);
    const auto output = reinterpret_cast<__gm__ uint64_t *>(packed);
    const uint32_t groups = k / group_k;
    constexpr uint32_t vectors = group_k / 16;
    const WriteSlices slices(packed, uint64_t(n) * k / 16, sizeof(uint64_t));
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t pos = slices.start(slice); pos < end; ++pos) {
            const uint64_t row = pos / vectors;
            const uint32_t channel = row / (tile_n * groups) * tile_n + row % tile_n;
            const uint32_t group = row / tile_n % groups;
            const uint64_t src = (uint64_t(channel) * groups + group) * vectors + pos % vectors;
            output[pos] = source[src];
        }
    }
    flush(packed);
}

extern "C" __global__ __aicore__ void plait_ascend_humming_pack(
    GM_ADDR activation, GM_ADDR weight, GM_ADDR scale, GM_ADDR zero, GM_ADDR packed_a, GM_ADDR packed_b,
    uint32_t rows, uint32_t padded_rows, uint32_t n, uint32_t k, uint32_t col_start,
    uint32_t cols, uint32_t depth, uint32_t first, uint32_t parts) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    plait_ascend::pack_projection_activation(
        activation, packed_a, rows, k, padded_rows, depth, depth, first, parts);
    pack_weights(weight, scale, zero, packed_b, n, k, col_start, cols, depth, first, parts);
}

extern "C" __global__ __aicore__ void plait_ascend_humming_finish(
    GM_ADDR dots, GM_ADDR sums, GM_ADDR output, uint32_t rows, uint32_t n,
    uint32_t col_start, uint32_t cols, uint32_t padded_rows, uint32_t first,
    uint32_t parts, uint32_t last, uint32_t output_bits) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t m_tile = PLAIT_ASCEND_TILE_M;
    constexpr uint32_t n_tile = PLAIT_ASCEND_TILE_N;
    const auto partials = reinterpret_cast<__gm__ const float *>(dots);
    const auto running = reinterpret_cast<__gm__ float *>(sums);
    const uint32_t column_tiles = cols / n_tile;
    const uint32_t tiles = padded_rows / m_tile * column_tiles;
    GM_ADDR dest = last ? output : sums;
    // Rows and column bands are 64 columns aligned. Mapping band-local
    // indices to output rows preserves the cache-line boundaries.
    const WriteSlices slices(dest, uint64_t(rows) * cols, last ? output_bits / 8 : sizeof(float));
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t index = slices.start(slice); index < end; ++index) {
            const uint32_t row = index / cols;
            const uint32_t col = index % cols;
            const uint64_t tile = uint64_t(row / m_tile) * column_tiles + col / n_tile;
            const uint32_t in_tile = (row % m_tile) * n_tile + col % n_tile;
            float sum = first ? running[index] : 0.0f;
            for (uint32_t part = 0; part < parts; ++part) {
                sum += partials[(uint64_t(part) * tiles + tile) * (m_tile * n_tile) + in_tile];
            }
            if (!last) {
                running[index] = sum;
                continue;
            }
            const uint64_t out_index = uint64_t(row) * n + col_start + col;
            if (output_bits == 32) reinterpret_cast<__gm__ float *>(output)[out_index] = sum;
            else reinterpret_cast<__gm__ half *>(output)[out_index] = half(sum);
        }
    }
    flush(dest);
}
