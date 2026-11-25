#pragma once

#include "w4a16.h"

#include <stdbool.h>
#include <stddef.h>

static inline uint32_t plait_ceil_div(uint32_t value, uint32_t divisor) {
    return value / divisor + (value % divisor != 0);
}

static inline bool plait_w4a16_valid_shape(const plait_w4a16_shape *shape) {
    if (!shape || !shape->m || !shape->n || !shape->k || !shape->group_size ||
        (shape->output_bits != 16 && shape->output_bits != 32)) {
        return false;
    }
    const uint64_t output = (uint64_t)shape->m * shape->n;
    return output <= UINT32_MAX && output <= SIZE_MAX / (shape->output_bits / 8) &&
           (uint64_t)shape->m * shape->k <= SIZE_MAX / 2 &&
           (uint64_t)shape->n * shape->k <= SIZE_MAX / 2;
}

static inline bool plait_w4_quantized_valid_shape(const plait_w4a16_shape *s, uint32_t bits) {
    return plait_w4a16_valid_shape(s) && (bits == 4 || bits == 8) &&
           (s->k < s->group_size ? s->k : s->group_size) <= INT32_MAX / (128 * 15);
}
