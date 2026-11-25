#include "w4_device_util.h"

using plait_ascend::WriteSlices;
using plait_ascend::flush;
using plait_ascend::group_count;

namespace {

__aicore__ inline float decode_f16(uint16_t bits) {
    const uint32_t magnitude = bits & 0x7fffu;
    const float value = magnitude < 0x400u ? float(magnitude) * 0x1p-24f :
        __builtin_bit_cast(float, (magnitude << 13) + 0x38000000u);
    return bits & 0x8000u ? -value : value;
}

template <int32_t Limit>
__aicore__ uint8_t encode(uint16_t bits, float scale) {
    if (scale == 0.0f) return 0;
    const float value = decode_f16(bits) / scale;
    const float magnitude = value < 0.0f ? -value : value;
    int32_t code = static_cast<int32_t>(magnitude);
    const float fraction = magnitude - float(code);
    if (fraction > 0.5f || (fraction == 0.5f && code % 2 != 0)) ++code;
    code = min(Limit, code);
    return static_cast<uint8_t>(value < 0.0f ? -code : code);
}

template <uint32_t Bits>
__aicore__ void pack_codes(GM_ADDR input, GM_ADDR scales, GM_ADDR codes,
                           uint32_t rows, uint32_t k, uint32_t group_size) {
    const auto values = reinterpret_cast<__gm__ const uint16_t *>(input);
    const auto scale = reinterpret_cast<__gm__ const float *>(scales);
    const auto out = reinterpret_cast<__gm__ uint8_t *>(codes);
    const uint32_t stride = Bits == 4 ? k / 2 + k % 2 : k;
    const uint32_t groups = group_count(k, group_size);
    const WriteSlices slices(codes, uint64_t(rows) * stride, 1, PLAIT_ASCEND_DCACHE_LINE_BYTES);
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t i = slices.start(slice); i < end; ++i) {
            const uint64_t row = i / stride;
            const auto a = values + row * k;
            const auto row_scales = scale + row * groups;
            if constexpr (Bits == 4) {
                const uint32_t col = (i % stride) * 2;
                const uint8_t lo = encode<7>(a[col], row_scales[col / group_size]) & 15;
                const uint8_t hi = col + 1 < k ?
                    encode<7>(a[col + 1], row_scales[(col + 1) / group_size]) & 15 : 0;
                out[i] = lo | (hi << 4);
            } else {
                const uint32_t col = i % stride;
                out[i] = encode<127>(a[col], row_scales[col / group_size]);
            }
        }
    }
    flush(codes);
}

}  // namespace

extern "C" __global__ __aicore__ void plait_ascend_quantize_scales(
    GM_ADDR input, GM_ADDR scales, uint32_t rows, uint32_t k, uint32_t group_size, uint32_t bits) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto values = reinterpret_cast<__gm__ const uint16_t *>(input);
    const auto scale = reinterpret_cast<__gm__ float *>(scales);
    const uint32_t groups = group_count(k, group_size);
    const float limit = bits == 4 ? 7.0f : 127.0f;
    const WriteSlices slices(scales, uint64_t(rows) * groups, sizeof(float), PLAIT_ASCEND_DCACHE_LINE_BYTES);
    for (uint64_t slice = AscendC::GetBlockIdx(); slices.start(slice) < slices.length;
         slice += AscendC::GetBlockNum()) {
        const uint64_t end = slices.end(slice);
        for (uint64_t i = slices.start(slice); i < end; ++i) {
            const uint32_t start = (i % groups) * group_size;
            const uint32_t count = min(group_size, k - start);
            const auto a = values + (i / groups) * k + start;
            uint32_t maximum = 0;
            for (uint32_t j = 0; j < count; ++j) maximum = max(maximum, uint32_t(a[j]) & 0x7fffu);
            scale[i] = maximum >= 0x7c00u ? 0.0f : maximum == 0 ? 1.0f :
                decode_f16(static_cast<uint16_t>(maximum)) / limit;
        }
    }
    flush(scales);
}

extern "C" __global__ __aicore__ void plait_ascend_quantize_pack(
    GM_ADDR input, GM_ADDR scales, GM_ADDR codes, GM_ADDR invalid,
    uint32_t rows, uint32_t k, uint32_t group_size, uint32_t bits) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (bits == 4) pack_codes<4>(input, scales, codes, rows, k, group_size);
    else pack_codes<8>(input, scales, codes, rows, k, group_size);

    // Only this core writes the status word. The first kernel reserves a zero scale
    // for non-finite inputs, distinct from any valid group including an all-zero group.
    if (AscendC::GetBlockIdx() == 0) {
        const auto scale = reinterpret_cast<__gm__ const float *>(scales);
        const uint64_t groups = uint64_t(rows) * group_count(k, group_size);
        uint32_t flag = 0;
        for (uint64_t group = 0; group < groups; ++group) {
            if (scale[group] == 0.0f) {
                flag = 1;
                break;
            }
        }
        *reinterpret_cast<__gm__ uint32_t *>(invalid) = flag;
        flush(invalid);
    }
}
