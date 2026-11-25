#include "w4_device_util.h"

using plait_ascend::WriteSlices;
using plait_ascend::flush;
using plait_ascend::group_count;

namespace {

constexpr uint32_t tile_m = PLAIT_ASCEND_TILE_M;
constexpr uint32_t tile_n = PLAIT_ASCEND_TILE_N;

__aicore__ inline int32_t nibble(__gm__ const uint8_t *row, uint32_t k) {
    return (row[k / 2] >> (4 * (k % 2))) & 15;
}

__aicore__ inline uint64_t expand_nibbles(uint32_t packed, uint64_t center) {
    uint64_t bytes = packed;
    bytes = (bytes | (bytes << 16)) & 0x0000ffff0000ffffULL;
    bytes = (bytes | (bytes << 8)) & 0x00ff00ff00ff00ffULL;
    constexpr uint64_t low = 0x000f000f000f000fULL;
    const uint64_t widened = (bytes & low) | ((bytes & (low << 4)) << 4);
    // The high bit in each byte prevents a zero-point subtraction from
    // borrowing into the adjacent lane. Toggling it yields signed INT8.
    constexpr uint64_t guard = 0x8080808080808080ULL;
    return ((widened | guard) - center) ^ guard;
}

template <uint32_t Bits, bool Weight>
__aicore__ void pack_words(__gm__ const uint8_t *source, __gm__ int8_t *destination,
                           uint32_t start, uint32_t count, uint32_t copied, int32_t center) {
    auto out = reinterpret_cast<__gm__ uint64_t *>(destination);
    uint32_t word = 0;
    const uint32_t words = copied / 8;
    if constexpr (Bits == 4) {
        auto in = reinterpret_cast<__gm__ const uint32_t *>(source) + start / 8;
        const uint64_t zero = uint64_t(center) * 0x0101010101010101ULL;
        for (; word < words; ++word) {
            const uint32_t codes = in[word];
            const uint32_t centered = Weight ? codes : codes ^ 0x88888888u;
            out[word] = expand_nibbles(centered, zero);
        }
    } else {
        auto in = reinterpret_cast<__gm__ const uint64_t *>(source) + start / 8;
        for (; word < words; ++word) out[word] = in[word];
    }
    for (; word < count / 8; ++word) out[word] = 0;
}

template <uint32_t Bits, bool Weight>
__aicore__ void pack(GM_ADDR input, GM_ADDR zero, GM_ADDR dest, uint32_t rows, uint32_t k,
                     uint32_t group_size, uint32_t padded_rows, uint32_t padded_k) {
    const auto source = reinterpret_cast<__gm__ const uint8_t *>(input);
    const auto zeros = reinterpret_cast<__gm__ const uint8_t *>(zero);
    const auto out = reinterpret_cast<__gm__ int8_t *>(dest);
    const uint32_t stride = Bits == 4 ? k / 2 + k % 2 : k;
    const uint32_t groups = group_count(k, group_size);
    const uint64_t group_stride = uint64_t(padded_rows) * padded_k;
    const WriteSlices slices(dest, group_stride * groups, 1);
    // Rows, groups and output slices must all preserve the word boundaries.
    const bool aligned = k % 8 == 0 && group_size % 8 == 0 &&
        reinterpret_cast<uint64_t>(input) % (Bits == 4 ? 4 : 8) == 0 &&
        reinterpret_cast<uint64_t>(dest) % 8 == 0;
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        uint64_t pos = slices.start(slice);
        const uint64_t end = slices.end(slice);
        while (pos < end) {
            const uint32_t group = pos / group_stride;
            const uint32_t row = (pos % group_stride) / padded_k;
            const uint32_t offset = pos % padded_k;
            const uint32_t begin = group * group_size;
            const uint32_t valid = min(group_size, k - begin);
            const uint32_t count = min(uint64_t(padded_k - offset), end - pos);
            if (row < rows && offset < valid) {
                const uint32_t copied = min(valid - offset, count);
                const auto from = source + uint64_t(row) * stride;
                const int32_t center = Weight ? zeros[uint64_t(row) * groups + group] : 8;
                if (aligned) {
                    pack_words<Bits, Weight>(from, out + pos, begin + offset, count, copied, center);
                } else {
                    for (uint32_t i = 0; i < copied; ++i) {
                        const uint32_t column = begin + offset + i;
                        if constexpr (Weight) out[pos + i] = nibble(from, column) - center;
                        else if constexpr (Bits == 4) out[pos + i] = (nibble(from, column) ^ 8) - 8;
                        else out[pos + i] = static_cast<int8_t>(from[column]);
                    }
                    for (uint32_t i = copied; i < count; ++i) out[pos + i] = 0;
                }
            } else if (aligned) {
                auto padding = reinterpret_cast<__gm__ uint64_t *>(out + pos);
                for (uint32_t i = 0; i < count / 8; ++i) padding[i] = 0;
            } else {
                for (uint32_t i = 0; i < count; ++i) out[pos + i] = 0;
            }
            pos += count;
        }
    }
    flush(dest);
}

template <uint32_t Bits>
__aicore__ int32_t scalar_dot(__gm__ const uint8_t *a, __gm__ const uint8_t *w,
                              uint32_t begin, uint32_t count, int32_t zero) {
    int32_t dot = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t j = begin + i;
        const int32_t x = Bits == 4 ? (nibble(a, j) ^ 8) - 8 : int32_t(a[j] ^ 128) - 128;
        dot += x * (nibble(w, j) - zero);
    }
    return dot;
}

}  // namespace

extern "C" __global__ __aicore__ void plait_ascend_pack_w4(
    GM_ADDR activation, GM_ADDR weight, GM_ADDR zero, GM_ADDR packed_a, GM_ADDR packed_b,
    uint32_t m, uint32_t n, uint32_t k, uint32_t group_size,
    uint32_t padded_m, uint32_t padded_n, uint32_t padded_k, uint32_t bits) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (bits == 4) {
        pack<4, false>(activation, zero, packed_a, m, k, group_size, padded_m, padded_k);
    } else {
        pack<8, false>(activation, zero, packed_a, m, k, group_size, padded_m, padded_k);
    }
    pack<4, true>(weight, zero, packed_b, n, k, group_size, padded_n, padded_k);
}

extern "C" __global__ __aicore__ void plait_ascend_finish_w4(
    GM_ADDR activation, GM_ADDR activation_scale, GM_ADDR weight, GM_ADDR weight_scale,
    GM_ADDR zero, GM_ADDR bias, GM_ADDR scratch, GM_ADDR output,
    uint32_t m, uint32_t n, uint32_t k, uint32_t group_size, uint32_t bits,
    uint32_t output_bits, uint32_t has_bias, uint32_t use_cube) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto a = reinterpret_cast<__gm__ const uint8_t *>(activation);
    const auto w = reinterpret_cast<__gm__ const uint8_t *>(weight);
    const auto as = reinterpret_cast<__gm__ const float *>(activation_scale);
    const auto ws = reinterpret_cast<__gm__ const half *>(weight_scale);
    const auto z = reinterpret_cast<__gm__ const uint8_t *>(zero);
    const auto b = reinterpret_cast<__gm__ const half *>(bias);
    const auto dots = reinterpret_cast<__gm__ const int32_t *>(scratch);
    const uint32_t groups = group_count(k, group_size);
    const uint32_t n_tiles = 1 + (n - 1) / tile_n;
    const uint64_t tiles_per_group = uint64_t(1 + (m - 1) / tile_m) * n_tiles;
    const uint32_t w_stride = k / 2 + k % 2;
    const uint32_t a_stride = bits == 4 ? w_stride : k;
    const WriteSlices slices(output, uint64_t(m) * n, output_bits / 8);
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t index = slices.start(slice); index < end; ++index) {
            const uint64_t row = index / n;
            const uint64_t col = index % n;
            const uint64_t tile = (row / tile_m) * n_tiles + col / tile_n;
            const uint32_t in_tile = (row % tile_m) * tile_n + col % tile_n;
            const auto ar = a + row * a_stride;
            const auto wr = w + col * w_stride;
            float sum = 0;
            for (uint32_t group = 0; group < groups; ++group) {
                const uint64_t parameter = col * groups + group;
                int32_t dot;
                if (use_cube) {
                    dot = dots[(group * tiles_per_group + tile) * (tile_m * tile_n) + in_tile];
                } else {
                    const uint32_t begin = group * group_size;
                    const uint32_t count = min(group_size, k - begin);
                    dot = bits == 4 ? scalar_dot<4>(ar, wr, begin, count, z[parameter])
                                    : scalar_dot<8>(ar, wr, begin, count, z[parameter]);
                }
                const float scale = as[row * groups + group] * float(ws[parameter]);
                sum += float(dot) * scale;
            }
            if (has_bias) sum += float(b[col]);
            if (output_bits == 32) reinterpret_cast<__gm__ float *>(output)[index] = sum;
            else reinterpret_cast<__gm__ half *>(output)[index] = half(sum);
        }
    }
    flush(output);
}
