#pragma once

#include "../w4a16.h"

#ifdef __cplusplus
extern "C" {
#endif

int plait_humming_probe(int ordinal, uint32_t n, uint32_t k, uint32_t group_size);
int plait_humming_repack(const void *canonical, void *packed, void *stream);
int plait_humming_launch(const plait_w4a16_shape *shape, const void *activation,
                         const void *weight, const void *scale, const void *zero,
                         void *output, void *stream);

#ifdef __cplusplus
}
#endif
