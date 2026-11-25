#pragma once

namespace plait_activation_quantizer {

constexpr uint32_t lanes = 32;
constexpr uint32_t threads = 128;

__device__ __forceinline__ uint32_t lane_max(uint32_t value, uint32_t distance) {
#if defined(__HIP__)
    return max(value, __shfl_xor(value, distance, lanes));
#else
    return max(value, __shfl_xor_sync(0xffffffffu, value, distance, lanes));
#endif
}

template <uint32_t Bits>
__device__ __forceinline__ uint8_t encode(__half value, float scale) {
    constexpr int limit = Bits == 4 ? 7 : 127;
    if (scale == 0.0f) return 0;
    const int rounded = __float2int_rn(__fdiv_rn(__half2float(value), scale));
    return uint8_t(max(-limit, min(limit, rounded)));
}

template <uint32_t Bits, bool Pack>
__global__ void group_scales(const __half *input, uint8_t *codes, float *scales, uint32_t *invalid,
                              uint32_t rows, uint32_t k, uint32_t group_size, uint32_t groups) {
    constexpr uint32_t groups_per_block = threads / lanes;
    constexpr float limit = Bits == 4 ? 7.0f : 127.0f;
    const uint32_t lane = threadIdx.x % lanes;
    const uint32_t packed_stride = Bits == 4 ? k / 2 + k % 2 : k;
    const uint64_t tasks = uint64_t(rows) * groups;
    for (uint64_t task = uint64_t(blockIdx.x) * groups_per_block + threadIdx.x / lanes;
         task < tasks; task += uint64_t(gridDim.x) * groups_per_block) {
        const uint64_t row = task / groups;
        const uint32_t start = (task % groups) * group_size;
        const uint32_t count = min(group_size, k - start);
        const auto source = input + row * k + start;
        uint32_t magnitude = 0;
        for (uint32_t j = lane; j < count; j += lanes) {
            magnitude = max(magnitude, uint32_t(__half_as_ushort(source[j])) & 0x7fffu);
        }
        for (uint32_t distance = lanes / 2; distance != 0; distance /= 2) {
            magnitude = lane_max(magnitude, distance);
        }
        const bool finite = magnitude < 0x7c00u;
        // Zero is reserved for a rejected group; a valid all-zero group has scale 1.
        const float scale = !finite ? 0.0f : magnitude == 0 ? 1.0f :
            __fdiv_rn(__half2float(__ushort_as_half(uint16_t(magnitude))), limit);
        if (lane == 0) {
            if (!finite) atomicExch(invalid, 1u);
            scales[task] = scale;
        }
        if constexpr (Pack && Bits == 8) {
            auto dest = codes + row * packed_stride + start;
            for (uint32_t j = lane; j < count; j += lanes) {
                dest[j] = encode<8>(source[j], scale);
            }
        }
        if constexpr (Pack && Bits == 4) {
            // Even-sized groups own their complete bytes, including any padded final nibble.
            auto dest = codes + row * packed_stride + start / 2;
            for (uint32_t j = lane * 2; j < count; j += lanes * 2) {
                const uint8_t lo = encode<4>(source[j], scale) & 15;
                const uint8_t hi = j + 1 < count ? encode<4>(source[j + 1], scale) & 15 : 0;
                dest[j / 2] = lo | (hi << 4);
            }
        }
    }
}

// With odd group lengths, one byte can contain codes from different groups.
// Compute the scales first, then assign the whole output byte to one thread.
__global__ void pack_odd_groups(const __half *input, const float *scales, uint8_t *codes,
                                 uint32_t rows, uint32_t k, uint32_t group_size, uint32_t groups) {
    const uint32_t stride = k / 2 + k % 2;
    for (uint64_t byte = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         byte < uint64_t(rows) * stride; byte += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = byte / stride;
        const uint32_t col = (byte % stride) * 2;
        const auto source = input + row * k;
        const auto group_scales = scales + row * groups;
        const uint8_t lo = encode<4>(source[col], group_scales[col / group_size]) & 15;
        const uint8_t hi = col + 1 < k ?
            encode<4>(source[col + 1], group_scales[(col + 1) / group_size]) & 15 : 0;
        codes[byte] = lo | (hi << 4);
    }
}

static uint32_t grid_for(uint32_t count, uint32_t per_block) {
    return min(plait_ceil_div(count, per_block), 65535u);
}

static PLAIT_GPU(Error_t) launch(PLAIT_GPU(Stream_t) stream, uint32_t rows, uint32_t k,
                                uint32_t group_size, uint32_t bits, const __half *input,
                                uint8_t *codes, float *scales, uint32_t *invalid) {
    const uint32_t groups = plait_ceil_div(k, group_size);
    const uint32_t grid = grid_for(rows * groups, threads / lanes);
    if (bits == 8) {
        group_scales<8, true><<<grid, threads, 0, stream>>>(input, codes, scales, invalid, rows, k, group_size, groups);
    } else if (group_size % 2 == 0) {
        group_scales<4, true><<<grid, threads, 0, stream>>>(input, codes, scales, invalid, rows, k, group_size, groups);
    } else {
        group_scales<4, false><<<grid, threads, 0, stream>>>(input, codes, scales, invalid, rows, k, group_size, groups);
        auto status = PLAIT_GPU(GetLastError)();
        if (status != PLAIT_GPU(Success)) return status;
        const uint32_t packed_grid = grid_for(rows * plait_ceil_div(k, 2), threads);
        pack_odd_groups<<<packed_grid, threads, 0, stream>>>(input, scales, codes, rows, k, group_size, groups);
    }
    return PLAIT_GPU(GetLastError)();
}

}  // namespace plait_activation_quantizer
