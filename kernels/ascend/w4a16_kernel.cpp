#include "kernel_operator.h"

extern "C" __global__ __aicore__ void plait_ascend_w4a16(
    GM_ADDR activation, GM_ADDR weight, GM_ADDR scale, GM_ADDR zero, GM_ADDR bias, GM_ADDR output,
    uint32_t m, uint32_t n, uint32_t k, uint32_t group_size, uint32_t output_bits,
    uint32_t has_bias) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto a = reinterpret_cast<__gm__ const half *>(activation);
    const auto w = reinterpret_cast<__gm__ const uint8_t *>(weight);
    const auto s = reinterpret_cast<__gm__ const half *>(scale);
    const auto z = reinterpret_cast<__gm__ const uint8_t *>(zero);
    const auto b = reinterpret_cast<__gm__ const half *>(bias);
    const uint32_t packed_stride = k / 2 + (k % 2 != 0);
    const uint32_t groups = k / group_size + (k % group_size != 0);
    // The single-core reference keeps scalar stores to each cache line ordered.
    for (uint32_t index = 0; index < m * n; ++index) {
        const uint64_t row = index / n;
        const uint64_t channel = index % n;
        float sum = 0;
        for (uint32_t inner = 0; inner < k; ++inner) {
            const uint8_t pair = w[channel * packed_stride + inner / 2];
            const int code = (pair >> ((inner % 2) * 4)) & 15;
            const uint64_t group = channel * groups + inner / group_size;
            const float weight_value = (code - int(z[group])) * float(s[group]);
            sum += float(a[row * k + inner]) * weight_value;
        }
        if (has_bias) sum += float(b[channel]);
        if (output_bits == 32) {
            reinterpret_cast<__gm__ float *>(output)[index] = sum;
        } else {
            reinterpret_cast<__gm__ half *>(output)[index] = half(sum);
        }
    }
    AscendC::GlobalTensor<uint64_t> written;
    written.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output));
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                     AscendC::DcciDst::CACHELINE_OUT>(written);
}
