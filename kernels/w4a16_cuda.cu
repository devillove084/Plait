#include "w4a16_layout.h"

// Compile the scalar reference with either CUDA/PPU or ROCm HIP.
#if defined(__HIP__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define PLAIT_GPU(name) hip##name
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define PLAIT_GPU(name) cuda##name
#endif

#if defined(__HIP__) && (defined(PLAIT_W4A16_MARLIN) || defined(PLAIT_W4A16_HUMMING))
#error "These Marlin and Humming builds require the CUDA compiler path"
#endif
#include <stdint.h>
#include <stddef.h>
#include <new>

#if defined(PLAIT_W4A16_MARLIN)
extern "C" int plait_marlin_launch(
    const uint16_t *activation, const uint32_t *packed_weight, uint16_t *scale,
    uint16_t *output, int32_t *workspace, int m, int n, int k, int group_size,
    int device, void *stream);
#if defined(PLAIT_W4_MARLIN_SHARDS)
extern "C" int plait_marlin_f32_launch(
    const uint16_t *activation, const uint32_t *packed_weight, uint16_t *scale,
    float *output, int32_t *workspace, int m, int n, int k, int group_size,
    int device, void *stream);
#if defined(PLAIT_W4_PPU_INTEGER_MMA)
size_t marlin_f32_workspace_bytes(uint32_t m, uint32_t n);
#endif
#endif
#endif

#if defined(PLAIT_W4A16_HUMMING)
#include "humming/plait_humming.h"
#endif

#if defined(__HIP__) && (defined(PLAIT_W4_HIP_MARLIN_MFMA) || defined(PLAIT_W4_HIP_HUMMING_MFMA))
#include <string.h>
static int probe_gfx942(int ordinal) {
    hipDeviceProp_t properties;
    const auto status = hipGetDeviceProperties(&properties, ordinal);
    if (status != hipSuccess) return status;
    const auto *arch = properties.gcnArchName;
    return strncmp(arch, "gfx942", 6) == 0 && (arch[6] == 0 || arch[6] == ':') ? 0 : -1;
}
#endif

#if defined(__HIP__) && defined(PLAIT_W4_HIP_MARLIN_MFMA)
#include "marlin/plait_marlin_hip.cuh"
#endif
#if defined(__HIP__) && defined(PLAIT_W4_HIP_HUMMING_MFMA)
#include "humming/plait_humming_hip.cuh"
#define PLAIT_W4_HUMMING_ABI_V2
#elif defined(PLAIT_W4A16_HUMMING)
#define PLAIT_W4_HUMMING_ABI_V2
#endif

#if defined(PLAIT_W4_CUDA_INTEGER_MMA) || defined(PLAIT_W4_PPU_INTEGER_MMA)
#include "w4_integer_mma.cuh"
#endif

#if (defined(__HIP__) && defined(PLAIT_W4_HIP_INTEGER_DOT)) || \
    defined(PLAIT_W4_CUDA_INTEGER_MMA) || defined(PLAIT_W4_PPU_INTEGER_MMA)
#define PLAIT_W4_INTEGER_ACTIVATION
#include "w4_quantize_f16.cuh"
#endif

namespace {

__global__ void w4a16(const __half *a, const uint8_t *w, const __half *scale, const uint8_t *zero,
                      const __half *bias, void *out, plait_w4a16_shape s,
                      uint32_t packed_stride, uint32_t group_count) {
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < uint64_t(s.m) * s.n; index += stride) {
        const auto row = index / s.n;
        const auto channel = index % s.n;
        float sum = 0;
        for (uint32_t inner = 0; inner < s.k; ++inner) {
            const uint8_t pair = w[channel * packed_stride + inner / 2];
            const int code = (pair >> ((inner % 2) * 4)) & 15;
            const auto group = channel * group_count + inner / s.group_size;
            const float weight = (code - int(zero[group])) * __half2float(scale[group]);
            sum = fmaf(__half2float(a[row * s.k + inner]), weight, sum);
        }
        if (bias) {
            sum += __half2float(bias[channel]);
        }
        if (s.output_bits == 32) {
            static_cast<float *>(out)[index] = sum;
        } else {
            static_cast<__half *>(out)[index] = __float2half_rn(sum);
        }
    }
}

#if defined(__HIP__) && defined(PLAIT_W4_HIP_INTEGER_DOT)
__device__ uint32_t load_packed32(const uint8_t *p) {
    return p[0] | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

__device__ int nibble(const uint8_t *row, uint32_t column) {
    return (row[column / 2] >> (4 * (column % 2))) & 15;
}

template <uint32_t Bits>
__device__ int integer_dot(const uint8_t *a, const uint8_t *w, uint32_t begin,
                           uint32_t end, int zero) {
    int sum = 0;
    uint32_t column = begin;
    if (column % 2 != 0) {
        int code = Bits == 4 ? (nibble(a, column) ^ 8) - 8 : int(a[column] ^ 128) - 128;
        sum += code * (nibble(w, column) - zero);
        ++column;
    }
    if constexpr (Bits == 4) {
        int activation_sum = 0;
        for (; end - column >= 8; column += 8) {
            const auto a8 = load_packed32(a + column / 2);
            const auto w8 = load_packed32(w + column / 2) ^ 0x88888888u;
            sum = __builtin_amdgcn_sdot8(a8, w8, sum, false);
            if (zero != 8) {
                activation_sum = __builtin_amdgcn_sdot8(a8, 0x11111111u, activation_sum, false);
            }
        }
        sum += (8 - zero) * activation_sum;
    } else {
        for (; end - column >= 4; column += 4) {
            const uint32_t weights = w[column / 2] | (uint32_t(w[column / 2 + 1]) << 8);
            uint32_t w4 = 0;
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                const int code = int((weights >> (4 * i)) & 15) - zero;
                w4 |= uint32_t(uint8_t(code)) << (8 * i);
            }
            sum = __builtin_amdgcn_sdot4(load_packed32(a + column), w4, sum, false);
        }
    }
    for (; column < end; ++column) {
        int code = Bits == 4 ? (nibble(a, column) ^ 8) - 8 : int(a[column] ^ 128) - 128;
        sum += code * (nibble(w, column) - zero);
    }
    return sum;
}

template <uint32_t Bits>
__global__ void w4_integer(const uint8_t *a, const float *as, const uint8_t *w,
                           const __half *ws, const uint8_t *zero, const __half *bias,
                           void *output, plait_w4a16_shape s, uint32_t groups,
                           uint32_t w_stride) {
    const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
    const uint32_t a_stride = Bits == 4 ? w_stride : s.k;
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < uint64_t(s.m) * s.n; index += step) {
        const uint64_t row = index / s.n;
        const uint64_t channel = index % s.n;
        const uint8_t *a_row = a + row * a_stride;
        const uint8_t *w_row = w + channel * w_stride;
        float sum = 0.0f;
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t begin = group * s.group_size;
            const uint32_t count = min(s.group_size, s.k - begin);
            const auto parameter = channel * groups + group;
            const int dot = integer_dot<Bits>(a_row, w_row, begin, begin + count, zero[parameter]);
            sum = fmaf(float(dot), as[row * groups + group] * __half2float(ws[parameter]), sum);
        }
        if (bias) sum += __half2float(bias[channel]);
        if (s.output_bits == 32) static_cast<float *>(output)[index] = sum;
        else static_cast<__half *>(output)[index] = __float2half_rn(sum);
    }
}
#endif

struct DeviceBuffer {
    void *ptr = nullptr;

    ~DeviceBuffer() {
        if (ptr) {
            (void)PLAIT_GPU(Free)(ptr);
        }
    }

    PLAIT_GPU(Error_t) upload(const void *source, size_t bytes) {
        auto status = PLAIT_GPU(Malloc)(&ptr, bytes);
        return status == PLAIT_GPU(Success) ? PLAIT_GPU(Memcpy)(ptr, source, bytes, PLAIT_GPU(MemcpyHostToDevice)) : status;
    }
};

}  // namespace

extern "C" int plait_w4a16_launch(const plait_w4a16_shape *shape,
                                   const uint16_t *a, const uint8_t *w, const uint16_t *scale, const uint8_t *zero,
                                   const uint16_t *bias, void *out, void *stream) {
    if (!plait_w4a16_valid_shape(shape) || !a || !w || !scale || !zero || !out) {
        return PLAIT_GPU(ErrorInvalidValue);
    }
    const auto s = *shape;
    constexpr uint32_t threads = 128;
    const auto blocks = plait_ceil_div(s.m * s.n, threads);
    w4a16<<<blocks > 65535 ? 65535 : blocks, threads, 0, static_cast<PLAIT_GPU(Stream_t)>(stream)>>>(
        reinterpret_cast<const __half *>(a), w, reinterpret_cast<const __half *>(scale), zero,
        reinterpret_cast<const __half *>(bias), out, s,
        plait_ceil_div(s.k, 2), plait_ceil_div(s.k, s.group_size));
    return PLAIT_GPU(GetLastError)();
}

extern "C" int plait_w4a16_run_host(const plait_w4a16_shape *shape, int ordinal,
                                     const uint16_t *a, const uint8_t *w, const uint16_t *scale, const uint8_t *zero,
                                     const uint16_t *bias, void *out) {
    if (!plait_w4a16_valid_shape(shape) || !a || !w || !scale || !zero || !out) {
        return PLAIT_GPU(ErrorInvalidValue);
    }
    const auto s = *shape;
    DeviceBuffer da, dw, ds, dz, db, dy;
    const size_t output_bytes = size_t(s.m) * s.n * (s.output_bits / 8);
    auto status = PLAIT_GPU(SetDevice)(ordinal);
    if (status != PLAIT_GPU(Success)) return status;
    status = da.upload(a, size_t(s.m) * s.k * 2);
    if (status != PLAIT_GPU(Success)) return status;
    status = dw.upload(w, size_t(s.n) * plait_ceil_div(s.k, 2));
    if (status != PLAIT_GPU(Success)) return status;
    const size_t group_elements = size_t(s.n) * plait_ceil_div(s.k, s.group_size);
    status = ds.upload(scale, group_elements * 2);
    if (status != PLAIT_GPU(Success)) return status;
    status = dz.upload(zero, group_elements);
    if (status != PLAIT_GPU(Success)) return status;
    if (bias) {
        status = db.upload(bias, size_t(s.n) * 2);
        if (status != PLAIT_GPU(Success)) return status;
    }
    status = PLAIT_GPU(Malloc)(&dy.ptr, output_bytes);
    if (status != PLAIT_GPU(Success)) return status;
    const auto launched = plait_w4a16_launch(&s, static_cast<const uint16_t *>(da.ptr),
        static_cast<const uint8_t *>(dw.ptr), static_cast<const uint16_t *>(ds.ptr), static_cast<const uint8_t *>(dz.ptr),
        static_cast<const uint16_t *>(db.ptr), dy.ptr, nullptr);
    if (launched != PLAIT_GPU(Success)) return launched;
    return PLAIT_GPU(Memcpy)(out, dy.ptr, output_bytes, PLAIT_GPU(MemcpyDeviceToHost));
}

namespace {

struct Session {
    int ordinal;
    PLAIT_GPU(Stream_t) stream;

    PLAIT_GPU(Error_t) bind() const { return PLAIT_GPU(SetDevice)(ordinal); }
};

int device_open(int ordinal, void **result) {
    auto status = PLAIT_GPU(SetDevice)(ordinal);
    if (status != PLAIT_GPU(Success)) return status;
    auto session = new (std::nothrow) Session{ordinal, nullptr};
    if (!session) return PLAIT_GPU(ErrorMemoryAllocation);
    status = PLAIT_GPU(StreamCreateWithFlags)(&session->stream, PLAIT_GPU(StreamNonBlocking));
    if (status != PLAIT_GPU(Success)) {
        delete session;
        return status;
    }
    *result = session;
    return PLAIT_GPU(Success);
}

void device_close(void *handle) {
    auto session = static_cast<Session *>(handle);
    if (session->bind() == PLAIT_GPU(Success)) {
        (void)PLAIT_GPU(StreamSynchronize)(session->stream);
        (void)PLAIT_GPU(StreamDestroy)(session->stream);
    }
    delete session;
}

int device_buffer_new(void *handle, const void *source, size_t bytes, void **buffer) {
    auto session = static_cast<Session *>(handle);
    auto status = session->bind();
    if (status != PLAIT_GPU(Success)) return status;
    void *allocation = nullptr;
    status = PLAIT_GPU(Malloc)(&allocation, bytes);
    if (status != PLAIT_GPU(Success)) return status;
    if (source) {
        status = PLAIT_GPU(MemcpyAsync)(allocation, source, bytes, PLAIT_GPU(MemcpyHostToDevice), session->stream);
        if (status == PLAIT_GPU(Success)) status = PLAIT_GPU(StreamSynchronize)(session->stream);
        if (status != PLAIT_GPU(Success)) {
            (void)PLAIT_GPU(Free)(allocation);
            return status;
        }
    }
    *buffer = allocation;
    return PLAIT_GPU(Success);
}

void device_buffer_free(void *handle, void *buffer) {
    auto session = static_cast<Session *>(handle);
    if (session->bind() == PLAIT_GPU(Success)) (void)PLAIT_GPU(Free)(buffer);
}

int device_buffer_read(void *handle, const void *buffer, void *destination, size_t bytes) {
    auto session = static_cast<Session *>(handle);
    auto status = session->bind();
    if (status != PLAIT_GPU(Success)) return status;
    return PLAIT_GPU(Memcpy)(destination, buffer, bytes, PLAIT_GPU(MemcpyDeviceToHost));
}

template <typename Submit>
int record_submission(Session *session, void **completion, Submit submit) {
    if (!session || !completion) return PLAIT_GPU(ErrorInvalidValue);
    auto status = session->bind();
    if (status != PLAIT_GPU(Success)) return status;
    PLAIT_GPU(Event_t) event = nullptr;
    status = PLAIT_GPU(EventCreateWithFlags)(&event, PLAIT_GPU(EventDisableTiming));
    if (status != PLAIT_GPU(Success)) return status;
    const int launched = submit();
    const int submitted = launched == PLAIT_GPU(Success) ? PLAIT_GPU(EventRecord)(event, session->stream) : launched;
    if (submitted != PLAIT_GPU(Success)) {
        (void)PLAIT_GPU(EventDestroy)(event);
        return submitted;
    }
    *completion = event;
    return PLAIT_GPU(Success);
}

int device_launch(void *handle, const plait_w4a16_shape *shape,
                  const void *activation, const void *weight, const void *scale, const void *zero,
                  const void *bias, void *output, void **completion) {
    auto session = static_cast<Session *>(handle);
    return record_submission(session, completion, [&] {
        return plait_w4a16_launch(
            shape, static_cast<const uint16_t *>(activation), static_cast<const uint8_t *>(weight),
            static_cast<const uint16_t *>(scale), static_cast<const uint8_t *>(zero),
            static_cast<const uint16_t *>(bias), output, session->stream);
    });
}

#if defined(PLAIT_W4A16_MARLIN)
int marlin_probe(void *handle) {
    auto session = static_cast<Session *>(handle);
    PLAIT_GPU(DeviceProp) properties;
    const auto status = PLAIT_GPU(GetDeviceProperties)(&properties, session->ordinal);
    if (status != PLAIT_GPU(Success)) return status;
#if defined(PLAIT_W4_PPU_INTEGER_MMA)
    constexpr int compiled_architecture = 80;
#else
    constexpr int compiled_architecture = 86;
#endif
    return properties.major * 10 + properties.minor == compiled_architecture ? 0 : -1;
}

template<typename Output, typename Launch>
int submit_marlin(void *handle, const plait_w4a16_shape *shape,
                        const void *activation, const void *weight, const void *scale,
                        void *workspace, void *output, void **completion, Launch launch) {
    if (!shape || shape->output_bits != sizeof(Output) * 8) return cudaErrorInvalidValue;
    auto session = static_cast<Session *>(handle);
    const auto s = *shape;
    return record_submission(session, completion, [&] {
#if defined(PLAIT_W4_PPU_INTEGER_MMA)
        constexpr bool needs_locks = sizeof(Output) == sizeof(uint16_t);
#else
        constexpr bool needs_locks = true;
#endif
        if constexpr (needs_locks) {
            const size_t workspace_bytes = size_t(s.n / 128) * 16 * sizeof(int32_t);
            const auto status = PLAIT_GPU(MemsetAsync)(workspace, 0, workspace_bytes, session->stream);
            if (status != PLAIT_GPU(Success)) return static_cast<int>(status);
        }
        if constexpr (sizeof(Output) == sizeof(uint16_t)) {
            const auto status = PLAIT_GPU(MemsetAsync)(output, 0, size_t(s.m) * s.n * sizeof(Output), session->stream);
            if (status != PLAIT_GPU(Success)) return static_cast<int>(status);
        }
        return launch(
            static_cast<const uint16_t *>(activation), static_cast<const uint32_t *>(weight),
            const_cast<uint16_t *>(static_cast<const uint16_t *>(scale)),
            static_cast<Output *>(output), static_cast<int32_t *>(workspace),
            s.m, s.n, s.k, s.group_size == s.k ? -1 : static_cast<int>(s.group_size),
            session->ordinal, session->stream);
    });
}

int device_launch_marlin(void *handle, const plait_w4a16_shape *shape,
                         const void *activation, const void *weight, const void *scale,
                         void *workspace, void *output, void **completion) {
    return submit_marlin<uint16_t>(handle, shape, activation, weight, scale, workspace,
                                   output, completion, plait_marlin_launch);
}

const plait_w4a16_marlin_api marlin_api = {marlin_probe, device_launch_marlin};
#endif

#if defined(__HIP__) && defined(PLAIT_W4_HIP_MARLIN_MFMA)
int marlin_mfma_probe(void *handle) {
    const auto session = static_cast<Session *>(handle);
    return probe_gfx942(session->ordinal);
}

int device_launch_marlin_mfma(void *handle, const plait_w4a16_shape *shape,
                              const void *activation, const void *weight, const void *params,
                              void *workspace, void *output, void **completion) {
    if (!plait_w4a16_valid_shape(shape) || shape->k % 16 != 0 || shape->n % 16 != 0 ||
        uint64_t(shape->k) * shape->n > INT32_MAX ||
        (shape->group_size != 128 && shape->group_size != shape->k) ||
        !activation || !weight || !params || !output || !completion) {
        return hipErrorInvalidValue;
    }
    auto session = static_cast<Session *>(handle);
    return record_submission(session, completion, [&] {
        return plait_marlin_mfma::launch(session->stream, *shape, activation, weight, params,
                                         workspace, output);
    });
}

const plait_w4a16_marlin_mfma_api marlin_mfma_api = {
    {marlin_mfma_probe, device_launch_marlin_mfma}, plait_marlin_mfma::workspace,
};
#endif

#if defined(PLAIT_W4_HUMMING_ABI_V2)
int submit_humming(void *handle, const plait_w4a16_shape *shape,
                   const void *activation, const void *weight, const void *scale,
                   const void *zero, void *output, void **completion,
                   decltype(&plait_humming_launch_shape) launch) {
    auto session = static_cast<Session *>(handle);
    return record_submission(session, completion, [&] {
        return launch(shape, activation, weight, scale, zero, output, session->stream);
    });
}
#endif

#if defined(PLAIT_W4A16_HUMMING)
int humming_probe(void *handle, uint32_t n, uint32_t k, uint32_t group_size) {
    const auto session = static_cast<Session *>(handle);
    return plait_humming_probe(session->ordinal, n, k, group_size);
}

int humming_repack(void *handle, const void *canonical, void *packed) {
    auto session = static_cast<Session *>(handle);
    const auto status = session->bind();
    if (status != cudaSuccess) return status;
    const int submitted = plait_humming_repack(canonical, packed, session->stream);
    return submitted == cudaSuccess ? cudaStreamSynchronize(session->stream) : submitted;
}

int device_launch_humming(void *handle, const plait_w4a16_shape *shape,
                         const void *activation, const void *weight, const void *scale,
                         const void *zero, void *output, void **completion) {
    if (!shape || shape->output_bits != 16) return cudaErrorInvalidValue;
    return submit_humming(handle, shape, activation, weight, scale, zero, output, completion,
                          plait_humming_launch);
}

const plait_w4a16_humming_api humming_api = {humming_probe, humming_repack, device_launch_humming};
#endif

#if defined(PLAIT_W4_HUMMING_ABI_V2)
int humming_probe_shape(void *handle, uint32_t n, uint32_t k, uint32_t group, uint32_t bits) {
    if (!handle) return PLAIT_GPU(ErrorInvalidValue);
    const auto session = static_cast<Session *>(handle);
    return plait_humming_probe_shape(session->ordinal, n, k, group, bits);
}

int humming_repack_shape(void *handle, uint32_t n, uint32_t k, uint32_t group,
                          const void *canonical, void *packed) {
    if (!handle) return PLAIT_GPU(ErrorInvalidValue);
    auto session = static_cast<Session *>(handle);
    const auto status = session->bind();
    if (status != PLAIT_GPU(Success)) return status;
    const int submitted = plait_humming_repack_shape(n, k, group, canonical, packed, session->stream);
    return submitted == PLAIT_GPU(Success) ? PLAIT_GPU(StreamSynchronize)(session->stream) : submitted;
}

int device_launch_humming_shape(void *handle, const plait_w4a16_shape *shape,
                               const void *activation, const void *weight, const void *scale,
                               const void *zero, void *output, void **completion) {
    if (!handle || !shape || !activation || !weight || !scale || !zero || !output || !completion) {
        return PLAIT_GPU(ErrorInvalidValue);
    }
    return submit_humming(handle, shape, activation, weight, scale, zero, output, completion,
                          plait_humming_launch_shape);
}

const plait_w4a16_humming_api_v2_table humming_api_v2 = {
    humming_probe_shape, humming_repack_shape, device_launch_humming_shape,
};
#endif

int device_wait(void *handle, void *completion) {
    auto session = static_cast<Session *>(handle);
    auto status = session->bind();
    if (status != PLAIT_GPU(Success)) return status;
    return PLAIT_GPU(EventSynchronize)(static_cast<PLAIT_GPU(Event_t)>(completion));
}

void device_completion_free(void *handle, void *completion) {
    auto session = static_cast<Session *>(handle);
    if (session->bind() == PLAIT_GPU(Success)) {
        (void)PLAIT_GPU(EventDestroy)(static_cast<PLAIT_GPU(Event_t)>(completion));
    }
}

const plait_w4a16_device_api api = {
    device_open, device_close, device_buffer_new, device_buffer_free,
    device_buffer_read, device_launch, device_wait, device_completion_free,
};

}  // namespace

extern "C" const plait_w4a16_device_api *plait_w4a16_device_api_v1(void) {
    return &api;
}

#if defined(__HIP__) && defined(PLAIT_W4_HIP_MARLIN_MFMA)
extern "C" const plait_w4a16_marlin_mfma_api *plait_w4a16_marlin_mfma_api_v1(void) {
    return &marlin_mfma_api;
}
#endif

#if defined(PLAIT_W4A16_MARLIN) && defined(PLAIT_W4_MARLIN_SHARDS)
#if defined(PLAIT_W4_PPU_INTEGER_MMA)
extern "C" size_t plait_w4a16_marlin_f32_workspace_v1(uint32_t rows, uint32_t columns) {
    return marlin_f32_workspace_bytes(rows, columns);
}
#endif

extern "C" int plait_w4a16_marlin_f32_launch_v1(
    void *handle, const plait_w4a16_shape *shape, const void *activation,
    const void *weight, const void *scale, void *workspace, void *output, void **completion) {
    return submit_marlin<float>(handle, shape, activation, weight, scale, workspace,
                                output, completion, plait_marlin_f32_launch);
}
#endif

#if defined(PLAIT_W4A16_HUMMING) && defined(PLAIT_W4_HUMMING_SHARDS)
extern "C" int plait_w4a16_humming_f32_launch_v1(
    void *handle, const plait_w4a16_shape *shape, const void *activation,
    const void *weight, const void *scale, const void *zero, void *output, void **completion) {
    if (!shape || shape->output_bits != 32 || !handle || !activation || !weight ||
        !scale || !zero || !output || !completion) return cudaErrorInvalidValue;
    return submit_humming(handle, shape, activation, weight, scale, zero, output, completion,
                          plait_humming_launch);
}
#endif

extern "C" int plait_w4_completion_query_v1(void *handle, void *completion, uint32_t *ready) {
    if (!handle || !completion || !ready) return PLAIT_GPU(ErrorInvalidValue);
    auto session = static_cast<Session *>(handle);
    auto status = session->bind();
    if (status != PLAIT_GPU(Success)) return status;
    status = PLAIT_GPU(EventQuery)(static_cast<PLAIT_GPU(Event_t)>(completion));
    if (status != PLAIT_GPU(Success) && status != PLAIT_GPU(ErrorNotReady)) return status;
    *ready = status == PLAIT_GPU(Success);
    return PLAIT_GPU(Success);
}

#if defined(PLAIT_W4_INTEGER_ACTIVATION)
extern "C" int plait_w4_quantize_f16_v1(
    void *handle, uint32_t rows, uint32_t columns, uint32_t group_size, uint32_t bits,
    const void *input, void *codes, void *scales, void *invalid, void **completion) {
    const plait_w4a16_shape shape{rows, 1, columns, group_size, 16};
    if (!plait_w4_quantized_valid_shape(&shape, bits) || uint64_t(rows) * columns > UINT32_MAX ||
        !handle || !input || !codes || !scales || !invalid || !completion) {
        return PLAIT_GPU(ErrorInvalidValue);
    }
    auto session = static_cast<Session *>(handle);
    return record_submission(session, completion, [&] {
        auto status = PLAIT_GPU(MemsetAsync)(invalid, 0, sizeof(uint32_t), session->stream);
        if (status != PLAIT_GPU(Success)) return status;
        return plait_activation_quantizer::launch(
            session->stream, rows, columns, group_size, bits, static_cast<const __half *>(input),
            static_cast<uint8_t *>(codes), static_cast<float *>(scales), static_cast<uint32_t *>(invalid));
    });
}

extern "C" int plait_w4_quantized_launch_v1(
    void *handle, const plait_w4a16_shape *shape, uint32_t bits, const void *activation,
    const void *activation_scale, const void *weight, const void *weight_scale, const void *zero,
    const void *bias, void *output, void **completion) {
    if (!plait_w4_quantized_valid_shape(shape, bits) || !activation || !activation_scale ||
        !weight || !weight_scale || !zero || !output) {
        return PLAIT_GPU(ErrorInvalidValue);
    }
    auto session = static_cast<Session *>(handle);
    const auto s = *shape;
    const auto groups = plait_ceil_div(s.k, s.group_size);
    const auto stride = plait_ceil_div(s.k, 2);
    return record_submission(session, completion, [&] {
        const auto a = static_cast<const uint8_t *>(activation);
        const auto as = static_cast<const float *>(activation_scale);
        const auto w = static_cast<const uint8_t *>(weight);
        const auto ws = static_cast<const __half *>(weight_scale);
        const auto z = static_cast<const uint8_t *>(zero);
        const auto b = static_cast<const __half *>(bias);
#if defined(__HIP__)
        constexpr uint32_t threads = 128;
        const auto blocks = plait_ceil_div(s.m * s.n, threads);
        const auto grid = blocks > 65535 ? 65535 : blocks;
        if (bits == 4) {
            w4_integer<4><<<grid, threads, 0, session->stream>>>(a, as, w, ws, z, b, output, s, groups, stride);
        } else {
            w4_integer<8><<<grid, threads, 0, session->stream>>>(a, as, w, ws, z, b, output, s, groups, stride);
        }
#else
        plait_integer_mma::launch(session->stream, bits, a, as, w, ws, z, b, output, s, groups, stride);
#endif
        return PLAIT_GPU(GetLastError)();
    });
}
#endif

#if defined(PLAIT_W4A16_MARLIN)
extern "C" const plait_w4a16_marlin_api *plait_w4a16_marlin_api_v1(void) {
    return &marlin_api;
}
#endif

#if defined(PLAIT_W4A16_HUMMING)
extern "C" const plait_w4a16_humming_api *plait_w4a16_humming_api_v1(void) {
    return &humming_api;
}
#endif

#if defined(PLAIT_W4_HUMMING_ABI_V2)
extern "C" const plait_w4a16_humming_api_v2_table *plait_w4a16_humming_api_v2(void) {
    return &humming_api_v2;
}
#endif
