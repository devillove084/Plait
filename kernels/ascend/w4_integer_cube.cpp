#include "w4_projection_cube.h"

extern "C" __global__ __aicore__ void plait_ascend_w4_cube(
    GM_ADDR packed_a, GM_ADDR packed_b, GM_ADDR scratch,
    uint32_t m, uint32_t n, uint32_t k, uint32_t groups) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    plait_ascend::project<int8_t, int32_t, PLAIT_ASCEND_CUBE_K, PLAIT_ASCEND_MAX_GROUP_K>(
        packed_a, packed_b, scratch, m, n, k, groups);
}

extern "C" __global__ __aicore__ void plait_ascend_marlin_cube(
    GM_ADDR packed_a, GM_ADDR packed_b, GM_ADDR scratch,
    uint32_t m, uint32_t n, uint32_t k, uint32_t groups) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    plait_ascend::project<half, float, 16, PLAIT_ASCEND_MARLIN_K_UNIT>(packed_a, packed_b, scratch, m, n, k, groups);
}
