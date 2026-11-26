#include "w4_humming_layout.h"
#include "w4_projection_cube.h"

extern "C" __global__ __aicore__ void plait_ascend_humming_cube(
    GM_ADDR packed_a, GM_ADDR packed_b, GM_ADDR scratch,
    uint32_t m, uint32_t n, uint32_t k, uint32_t parts) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    plait_ascend::project<half, float, 16, PLAIT_ASCEND_HUMMING_DEPTH>(
        packed_a, packed_b, scratch, m, n, k, parts);
}
