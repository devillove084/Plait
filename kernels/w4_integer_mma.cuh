#pragma once

namespace plait_integer_mma {

constexpr uint32_t warp_size = 32;
constexpr uint32_t threads = 128;
constexpr uint32_t warps_per_block = threads / warp_size;
constexpr unsigned full_warp = 0xffffffffu;

#if defined(PLAIT_W4_CUDA_INTEGER_MMA)
constexpr bool native_i4 = true;
#else
constexpr bool native_i4 = false;
#endif

// Load up to one register from a logical row. Mask the group tail before it reaches MMA.
template <uint32_t Bits, uint32_t Values>
__device__ __forceinline__ uint32_t load_codes(const uint8_t *row, uint32_t begin,
                                               uint32_t count, uint32_t offset) {
    if (offset >= count) return 0;
    const uint32_t column = begin + offset;
    uint32_t value = 0;
    if ((Bits == 8 || column % 2 == 0) && count - offset >= Values) {
        const uint8_t *p = row + column / (8 / Bits);
        #pragma unroll
        for (uint32_t i = 0; i < Values * Bits / 8; ++i) {
            value |= uint32_t(p[i]) << (i * 8);
        }
    } else {
        #pragma unroll
        for (uint32_t i = 0; i < Values; ++i) {
            if (i < count - offset) {
                const uint32_t k = column + i;
                const auto code = Bits == 4 ? (row[k / 2] >> (4 * (k % 2))) & 15 : row[k];
                value |= uint32_t(code) << (i * Bits);
            }
        }
    }
    return value;
}

__device__ __forceinline__ uint32_t widen4(uint32_t packed) {
    return (packed & 0x000fu) | ((packed & 0x00f0u) << 4) |
           ((packed & 0x0f00u) << 8) | ((packed & 0xf000u) << 12);
}

__device__ __forceinline__ uint32_t subtract_bytes(uint32_t codes, uint8_t zero) {
    // The guard bit prevents the subtraction in each lane borrowing from its neighbor.
    return ((codes | 0x80808080u) - uint32_t(zero) * 0x01010101u) ^ 0x80808080u;
}

template <uint32_t Bits>
__device__ __forceinline__ uint32_t activation_i8(const uint8_t *row, uint32_t begin,
                                                   uint32_t count, uint32_t offset) {
    const auto codes = load_codes<Bits, 4>(row, begin, count, offset);
    if constexpr (Bits == 4) return subtract_bytes(widen4(codes ^ 0x8888u), 8);
    return codes;
}

__device__ __forceinline__ uint32_t weight_i8(const uint8_t *row, uint32_t begin,
                                              uint32_t count, uint32_t offset, uint8_t zero) {
    return subtract_bytes(widen4(load_codes<4, 4>(row, begin, count, offset)), zero);
}

__device__ __forceinline__ int sum_signed_nibbles(uint32_t packed) {
    const auto biased = packed ^ 0x88888888u;
    return __dp4a(int(biased & 0x0f0f0f0fu), 0x01010101, -64) +
           __dp4a(int((biased >> 4) & 0x0f0f0f0fu), 0x01010101, 0);
}

__device__ __forceinline__ int sum_four_lanes(int value) {
    value += __shfl_xor_sync(full_warp, value, 1, 4);
    return value + __shfl_xor_sync(full_warp, value, 2, 4);
}

__device__ __forceinline__ void write(void *output, uint32_t bits, uint64_t index,
                                      float value, const __half *bias, uint32_t channel) {
    if (bias) value += __half2float(bias[channel]);
    if (bits == 32) static_cast<float *>(output)[index] = value;
    else static_cast<__half *>(output)[index] = __float2half_rn(value);
}

template <uint32_t Bits>
__global__ void dot_kernel(const uint8_t *a, const float *as, const uint8_t *w, const __half *ws,
                           const uint8_t *z, const __half *bias, void *output, plait_w4a16_shape s,
                           uint32_t groups, uint32_t packed_stride) {
    const uint32_t lane = threadIdx.x % warp_size;
    const uint32_t a_stride = Bits == 4 ? packed_stride : s.k;
    const uint64_t outputs = uint64_t(s.m) * s.n;
    const uint64_t step = uint64_t(gridDim.x) * warps_per_block;
    for (uint64_t index = uint64_t(blockIdx.x) * warps_per_block + threadIdx.x / warp_size;
         index < outputs; index += step) {
        const auto row = index / s.n;
        const auto channel = index % s.n;
        const auto ar = a + row * a_stride;
        const auto wr = w + channel * packed_stride;
        float result = 0;
        for (uint32_t group = 0; group < groups; ++group) {
            const auto weight_index = channel * groups + group;
            const uint32_t begin = group * s.group_size;
            const uint32_t count = min(s.group_size, s.k - begin);
            int dot = 0;
            for (uint32_t offset = lane * 4; offset < count; offset += warp_size * 4) {
                const auto x = activation_i8<Bits>(ar, begin, count, offset);
                const auto y = weight_i8(wr, begin, count, offset, z[weight_index]);
                dot = __dp4a(int(x), int(y), dot);
            }
            for (uint32_t distance = warp_size / 2; distance != 0; distance /= 2) {
                dot += __shfl_down_sync(full_warp, dot, distance);
            }
            if (lane == 0) {
                result = fmaf(float(dot), as[row * groups + group] * __half2float(ws[weight_index]), result);
            }
        }
        if (lane == 0) write(output, s.output_bits, index, result, bias, channel);
    }
}

template <uint32_t Bits>
__global__ void mma_kernel(const uint8_t *a, const float *as, const uint8_t *w, const __half *ws,
                           const uint8_t *z, const __half *bias, void *output, plait_w4a16_shape s,
                           uint32_t groups, uint32_t packed_stride) {
    // PTX m16n8 integer fragments map each four-lane group to two rows and one B column.
    constexpr bool i4 = Bits == 4 && native_i4;
    constexpr uint32_t k_tile = i4 ? 64 : 32;
    constexpr uint32_t fragment = k_tile / 8;
    const uint32_t lane = threadIdx.x % warp_size;
    const uint32_t lane_group = lane / 4;
    const uint32_t lane_in_group = lane % 4;
    const uint32_t a_stride = Bits == 4 ? packed_stride : s.k;
    const uint32_t n_tiles = 1 + (s.n - 1) / 8;
    const uint64_t tiles = uint64_t(1 + (s.m - 1) / 16) * n_tiles;
    const uint64_t step = uint64_t(gridDim.x) * warps_per_block;
    for (uint64_t tile = uint64_t(blockIdx.x) * warps_per_block + threadIdx.x / warp_size;
         tile < tiles; tile += step) {
        const uint32_t row0 = (tile / n_tiles) * 16 + lane_group;
        const uint32_t row1 = row0 + 8;
        const uint32_t col = (tile % n_tiles) * 8;
        const uint32_t weight_col = col + lane_group;
        const uint32_t out_col = col + 2 * lane_in_group;
        const auto a0 = row0 < s.m ? a + uint64_t(row0) * a_stride : nullptr;
        const auto a1 = row1 < s.m ? a + uint64_t(row1) * a_stride : nullptr;
        const auto wb = weight_col < s.n ? w + uint64_t(weight_col) * packed_stride : nullptr;
        float result[4]{};
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t begin = group * s.group_size;
            const uint32_t count = min(s.group_size, s.k - begin);
            const uint8_t zero = !i4 && wb ? z[uint64_t(weight_col) * groups + group] : 8;
            int dot[4]{};
            int activation_sum[2]{};
            for (uint32_t kk = 0; kk < count; kk += k_tile) {
                const uint32_t lo = kk + lane_in_group * fragment;
                const uint32_t hi = lo + k_tile / 2;
                uint32_t x[4], y[2];
                if constexpr (i4) {
                    x[0] = a0 ? load_codes<4, 8>(a0, begin, count, lo) : 0;
                    x[1] = a1 ? load_codes<4, 8>(a1, begin, count, lo) : 0;
                    x[2] = a0 ? load_codes<4, 8>(a0, begin, count, hi) : 0;
                    x[3] = a1 ? load_codes<4, 8>(a1, begin, count, hi) : 0;
                    y[0] = wb ? load_codes<4, 8>(wb, begin, count, lo) ^ 0x88888888u : 0;
                    y[1] = wb ? load_codes<4, 8>(wb, begin, count, hi) ^ 0x88888888u : 0;
                    activation_sum[0] += sum_signed_nibbles(x[0]) + sum_signed_nibbles(x[2]);
                    activation_sum[1] += sum_signed_nibbles(x[1]) + sum_signed_nibbles(x[3]);
                    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(dot[0]), "+r"(dot[1]), "+r"(dot[2]), "+r"(dot[3])
                        : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(y[0]), "r"(y[1]));
                } else {
                    x[0] = a0 ? activation_i8<Bits>(a0, begin, count, lo) : 0;
                    x[1] = a1 ? activation_i8<Bits>(a1, begin, count, lo) : 0;
                    x[2] = a0 ? activation_i8<Bits>(a0, begin, count, hi) : 0;
                    x[3] = a1 ? activation_i8<Bits>(a1, begin, count, hi) : 0;
                    y[0] = wb ? weight_i8(wb, begin, count, lo, zero) : 0;
                    y[1] = wb ? weight_i8(wb, begin, count, hi, zero) : 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(dot[0]), "+r"(dot[1]), "+r"(dot[2]), "+r"(dot[3])
                        : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(y[0]), "r"(y[1]));
                }
            }
            if constexpr (i4) {
                activation_sum[0] = sum_four_lanes(activation_sum[0]);
                activation_sum[1] = sum_four_lanes(activation_sum[1]);
            }
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                const auto row = i < 2 ? row0 : row1;
                const auto channel = out_col + i % 2;
                if (row < s.m && channel < s.n) {
                    const auto param = uint64_t(channel) * groups + group;
                    if constexpr (i4) dot[i] += (8 - int(z[param])) * activation_sum[i / 2];
                    result[i] = fmaf(float(dot[i]), as[uint64_t(row) * groups + group] *
                                    __half2float(ws[param]), result[i]);
                }
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            const auto row = i < 2 ? row0 : row1;
            const auto channel = out_col + i % 2;
            if (row < s.m && channel < s.n) {
                write(output, s.output_bits, uint64_t(row) * s.n + channel, result[i], bias, channel);
            }
        }
    }
}

void launch(cudaStream_t stream, uint32_t bits, const uint8_t *a, const float *as, const uint8_t *w,
            const __half *ws, const uint8_t *z, const __half *bias, void *out, plait_w4a16_shape s,
            uint32_t groups, uint32_t packed_stride) {
    const bool use_dot = s.m <= 4 && !(bits == 4 && native_i4);
    const uint32_t work = use_dot ? s.m * s.n : plait_ceil_div(s.m, 16) * plait_ceil_div(s.n, 8);
    const uint32_t blocks = plait_ceil_div(work, warps_per_block);
    const uint32_t grid = blocks > 65535 ? 65535 : blocks;
    if (use_dot) {
        if (bits == 4) dot_kernel<4><<<grid, threads, 0, stream>>>(a, as, w, ws, z, bias, out, s, groups, packed_stride);
        else dot_kernel<8><<<grid, threads, 0, stream>>>(a, as, w, ws, z, bias, out, s, groups, packed_stride);
    } else {
        if (bits == 4) mma_kernel<4><<<grid, threads, 0, stream>>>(a, as, w, ws, z, bias, out, s, groups, packed_stride);
        else mma_kernel<8><<<grid, threads, 0, stream>>>(a, as, w, ws, z, bias, out, s, groups, packed_stride);
    }
}

}  // namespace plait_integer_mma
