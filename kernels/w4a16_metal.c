#include "w4a16_layout.h"

#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

/* Metal exposes Objective-C objects. This C adapter uses their public ABI. */
extern id MTLCreateSystemDefaultDevice(void);
typedef unsigned long metal_index;  /* NSUInteger on Apple Silicon. */
typedef struct { metal_index width, height, depth; } metal_size;
typedef struct { metal_index location, length; } metal_range;

#define SELECTOR(name) sel_registerName(name)
#define MESSAGE(type) ((type)objc_msgSend)

static const char source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Shape { uint m, n, k, group_size, output_bits; };\n"
    "kernel void w4a16(device const half *a [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
    "                  device const half *scale [[buffer(2)]], device const uchar *zero [[buffer(3)]],\n"
    "                  device const half *bias [[buffer(4)]], device uchar *out [[buffer(5)]],\n"
    "                  constant Shape &s [[buffer(6)]], constant uint &has_bias [[buffer(7)]],\n"
    "                  uint index [[thread_position_in_grid]]) {\n"
    "    uint m = s.m, n = s.n, k = s.k, group_size = s.group_size;\n"
    "    if (index >= m * n) return;\n"
    "    ulong row = index / n, channel = index % n;\n"
    "    uint stride = k / 2 + (k % 2 != 0);\n"
    "    uint groups = k / group_size + (k % group_size != 0);\n"
    "    float sum = 0;\n"
    "    for (uint i = 0; i < k; ++i) {\n"
    "        uchar pair = w[channel * stride + i / 2];\n"
    "        int code = (pair >> ((i % 2) * 4)) & 15;\n"
    "        ulong group = channel * groups + i / group_size;\n"
    "        float weight = float(code - int(zero[group])) * float(scale[group]);\n"
    "        sum = fma(float(a[row * k + i]), weight, sum);\n"
    "    }\n"
    "    if (has_bias) sum += float(bias[channel]);\n"
    "    if (s.output_bits == 32) ((device float *)out)[index] = sum;\n"
    "    else ((device half *)out)[index] = half(sum);\n"
    "}\n"
    "inline int nibble(device const uchar *row, uint column) {\n"
    "    return (row[column / 2] >> ((column % 2) * 4)) & 15;\n"
    "}\n"
    "inline int activation_code(device const uchar *row, uint column, uint bits) {\n"
    "    return bits == 4 ? (nibble(row, column) ^ 8) - 8 : int(row[column] ^ 128) - 128;\n"
    "}\n"
    "inline int integer_dot(device const uchar *a, device const uchar *w,\n"
    "                       uint begin, uint end, uint bits, int zero) {\n"
    "    int sum = 0;\n"
    "    uint column = begin;\n"
    "    if (column % 2 != 0) {\n"
    "        sum += activation_code(a, column, bits) * (nibble(w, column) - zero);\n"
    "        ++column;\n"
    "    }\n"
    "    for (; end - column >= 4; column += 4) {\n"
    "        uint weights = w[column / 2] | (uint(w[column / 2 + 1]) << 8);\n"
    "        int4 wc = int4((uint4(weights) >> uint4(0, 4, 8, 12)) & 15) - zero;\n"
    "        int4 ac;\n"
    "        if (bits == 4) {\n"
    "            uint codes = a[column / 2] | (uint(a[column / 2 + 1]) << 8);\n"
    "            ac = int4(((uint4(codes) >> uint4(0, 4, 8, 12)) & 15) ^ 8) - 8;\n"
    "        } else {\n"
    "            ac = int4(uint4(a[column], a[column + 1], a[column + 2], a[column + 3]) ^ 128) - 128;\n"
    "        }\n"
    "        int4 products = ac * wc;\n"
    "        sum += products.x + products.y + products.z + products.w;\n"
    "    }\n"
    "    for (; column < end; ++column)\n"
    "        sum += activation_code(a, column, bits) * (nibble(w, column) - zero);\n"
    "    return sum;\n"
    "}\n"
    "kernel void w4_integer(device const uchar *a [[buffer(0)]], device const float *as [[buffer(1)]],\n"
    "                       device const uchar *w [[buffer(2)]], device const half *ws [[buffer(3)]],\n"
    "                       device const uchar *zero [[buffer(4)]], device const half *bias [[buffer(5)]],\n"
    "                       device uchar *out [[buffer(6)]], constant Shape &s [[buffer(7)]],\n"
    "                       constant uint2 &options [[buffer(8)]], uint index [[thread_position_in_grid]]) {\n"
    "    if (index >= s.m * s.n) return;\n"
    "    ulong row = index / s.n, channel = index % s.n;\n"
    "    uint bits = options.y;\n"
    "    uint w_stride = s.k / 2 + (s.k % 2 != 0);\n"
    "    uint a_stride = bits == 4 ? w_stride : s.k;\n"
    "    uint groups = s.k / s.group_size + (s.k % s.group_size != 0);\n"
    "    float sum = 0;\n"
    "    for (uint group = 0; group < groups; ++group) {\n"
    "        uint begin = group * s.group_size;\n"
    "        uint end = begin + min(s.group_size, s.k - begin);\n"
    "        ulong parameter = channel * groups + group;\n"
    "        int dot = integer_dot(a + row * a_stride, w + channel * w_stride, begin, end, bits, zero[parameter]);\n"
    "        sum = fma(float(dot), as[row * groups + group] * float(ws[parameter]), sum);\n"
    "    }\n"
    "    if (options.x) sum += float(bias[channel]);\n"
    "    if (s.output_bits == 32) ((device float *)out)[index] = sum;\n"
    "    else ((device half *)out)[index] = half(sum);\n"
    "}\n"
    "template <uint Bits>\n"
    "inline float simd_w4_dot(device const uchar *a, device const float *as, device const uchar *w,\n"
    "                        device const half *ws, device const uchar *zero,\n"
    "                        constant Shape &s, ulong row, ulong channel, uint lane) {\n"
    "    const uint groups = 1 + (s.k - 1) / s.group_size;\n"
    "    const uint stride = Bits == 4 ? s.k / 2 : s.k;\n"
    "    device const uchar *arow = a + row * stride;\n"
    "    device const ushort *wrow = (device const ushort *)(w + channel * (s.k / 2));\n"
    "    float sum = 0;\n"
    "    for (uint group = 0; group < groups; ++group) {\n"
    "        uint begin = group * s.group_size;\n"
    "        uint end = begin + min(s.group_size, s.k - begin);\n"
    "        ulong parameter = channel * groups + group;\n"
    "        int offset = zero[parameter];\n"
    "        int4 products = int4(0);\n"
    "        for (uint i = begin + lane * 4; i < end; i += 32) {\n"
    "            uint weights = wrow[i / 4];\n"
    "            int4 wc = int4((uint4(weights) >> uint4(0, 4, 8, 12)) & 15) - offset;\n"
    "            int4 ac;\n"
    "            if (Bits == 4) {\n"
    "                uint codes = ((device const ushort *)arow)[i / 4];\n"
    "                ac = int4(((uint4(codes) >> uint4(0, 4, 8, 12)) & 15) ^ 8) - 8;\n"
    "            } else {\n"
    "                uint codes = ((device const uint *)arow)[i / 4];\n"
    "                ac = int4(((uint4(codes) >> uint4(0, 8, 16, 24)) & 255) ^ 128) - 128;\n"
    "            }\n"
    "            products += ac * wc;\n"
    "        }\n"
    "        int dot = products.x + products.y + products.z + products.w;\n"
    "        dot += simd_shuffle_xor(dot, 4);\n"
    "        dot += simd_shuffle_xor(dot, 2);\n"
    "        dot += simd_shuffle_xor(dot, 1);\n"
    "        sum = fma(float(dot), as[row * groups + group] * float(ws[parameter]), sum);\n"
    "    }\n"
    "    return sum;\n"
    "}\n"
    "kernel void w4_integer_simd(device const uchar *a [[buffer(0)]], device const float *as [[buffer(1)]],\n"
    "                            device const uchar *w [[buffer(2)]], device const half *ws [[buffer(3)]],\n"
    "                            device const uchar *zero [[buffer(4)]], device const half *bias [[buffer(5)]],\n"
    "                            device uchar *out [[buffer(6)]], constant Shape &s [[buffer(7)]],\n"
    "                            constant uint2 &options [[buffer(8)]], uint task [[threadgroup_position_in_grid]],\n"
    "                            uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {\n"
    "    ulong index = ulong(task) * (width / 8) + lane / 8;\n"
    "    if (index >= ulong(s.m) * s.n) return;\n"
    "    ulong row = index / s.n, channel = index % s.n;\n"
    "    float sum = options.y == 4 ? simd_w4_dot<4>(a, as, w, ws, zero, s, row, channel, lane % 8)\n"
    "                              : simd_w4_dot<8>(a, as, w, ws, zero, s, row, channel, lane % 8);\n"
    "    if (lane % 8 != 0) return;\n"
    "    if (options.x) sum += float(bias[channel]);\n"
    "    if (s.output_bits == 32) ((device float *)out)[index] = sum;\n"
    "    else ((device half *)out)[index] = half(sum);\n"
    "}\n"
#include "w4_quantize_metal.h"
    ;

static struct {
    id device, queue, pipeline, integer_pipeline, integer_simd_pipeline, quantize_pipeline, pack_pipeline;
    metal_index integer_simd_width;
} metal;
static pthread_once_t initialized = PTHREAD_ONCE_INIT;

static id object(id instance, const char *name) {
    return MESSAGE(id (*)(id, SEL))(instance, SELECTOR(name));
}

static void invoke(id instance, const char *name) {
    MESSAGE(void (*)(id, SEL))(instance, SELECTOR(name));
}

static id autorelease_pool(void) {
    return object(object((id)objc_getClass("NSAutoreleasePool"), "alloc"), "init");
}

static id string(const char *value) {
    return MESSAGE(id (*)(id, SEL, const char *))(
        (id)objc_getClass("NSString"), SELECTOR("stringWithUTF8String:"), value);
}

static void show_error(id error) {
    const char *message = MESSAGE(const char *(*)(id, SEL))(
        object(error, "localizedDescription"), SELECTOR("UTF8String"));
    if (message) fprintf(stderr, "Metal W4A16: %s\n", message);
}

static id make_pipeline(id device, id library, const char *name) {
    id function = MESSAGE(id (*)(id, SEL, id))(
        library, SELECTOR("newFunctionWithName:"), string(name));
    id error = nil;
    id pipeline = function ? MESSAGE(id (*)(id, SEL, id, id *))(
        device, SELECTOR("newComputePipelineStateWithFunction:error:"), function, &error) : nil;
    if (!pipeline) show_error(error);
    invoke(function, "release");
    return pipeline;
}

static void initialize(void) {
    id pool = autorelease_pool();
    id device = MTLCreateSystemDefaultDevice();
    id queue = object(device, "newCommandQueue");
    id options = object((id)objc_getClass("MTLCompileOptions"), "new");
    MESSAGE(void (*)(id, SEL, BOOL))(options, SELECTOR("setFastMathEnabled:"), NO);
    id error = nil;
    id library = MESSAGE(id (*)(id, SEL, id, id, id *))(
        device, SELECTOR("newLibraryWithSource:options:error:"), string(source), options, &error);
    id pipeline = make_pipeline(device, library, "w4a16");
    if (device && queue && pipeline) {
        metal.device = device;
        metal.queue = queue;
        metal.pipeline = pipeline;
        metal.integer_pipeline = make_pipeline(device, library, "w4_integer");
        metal.integer_simd_pipeline = make_pipeline(device, library, "w4_integer_simd");
        metal.integer_simd_width = MESSAGE(metal_index (*)(id, SEL))(
            metal.integer_simd_pipeline, SELECTOR("threadExecutionWidth"));
        metal.quantize_pipeline = make_pipeline(device, library, "quantize_f16");
        metal.pack_pipeline = make_pipeline(device, library, "pack_odd_groups");
    } else {
        show_error(error);
        invoke(pipeline, "release");
        invoke(queue, "release");
        invoke(device, "release");
    }
    invoke(library, "release");
    invoke(options, "release");
    invoke(pool, "drain");
}

static int device_open(int ordinal, void **result) {
    if (ordinal != 0 || pthread_once(&initialized, initialize) || !metal.pipeline) return 2;
    id queue = object(metal.device, "newCommandQueue");
    if (!queue) return 3;
    *result = queue;
    return 0;
}

static void device_close(void *handle) {
    // Committed command buffers retain their resources until execution finishes.
    invoke((id)handle, "release");
}

static int device_buffer_new(void *handle, const void *source_bytes, size_t bytes, void **result) {
    (void)handle;
    // MTLResourceStorageModeShared.
    id buffer = source_bytes
        ? MESSAGE(id (*)(id, SEL, const void *, metal_index, metal_index))(
              metal.device, SELECTOR("newBufferWithBytes:length:options:"), source_bytes, bytes, 0)
        : MESSAGE(id (*)(id, SEL, metal_index, metal_index))(
              metal.device, SELECTOR("newBufferWithLength:options:"), bytes, 0);
    if (!buffer) return 3;
    *result = buffer;
    return 0;
}

static void device_buffer_free(void *handle, void *buffer) {
    (void)handle;
    invoke((id)buffer, "release");
}

static int device_buffer_read(void *handle, const void *buffer, void *destination, size_t bytes) {
    (void)handle;
    const void *contents = MESSAGE(void *(*)(id, SEL))((id)buffer, SELECTOR("contents"));
    if (!contents) return 3;
    memcpy(destination, contents, bytes);
    return 0;
}

static int encode_kernel(id command, id pipeline, const plait_w4a16_shape *s,
                         const id *buffers, metal_index count, const void *options,
                         size_t options_bytes, metal_index work, bool work_is_groups) {
    id encoder = object(command, "computeCommandEncoder");
    if (!encoder) return 3;
    MESSAGE(void (*)(id, SEL, id))(encoder, SELECTOR("setComputePipelineState:"), pipeline);
    for (metal_index i = 0; i < count; ++i) {
        MESSAGE(void (*)(id, SEL, id, metal_index, metal_index))(
            encoder, SELECTOR("setBuffer:offset:atIndex:"), buffers[i], 0, i);
    }
    MESSAGE(void (*)(id, SEL, const void *, metal_index, metal_index))(
        encoder, SELECTOR("setBytes:length:atIndex:"), s, sizeof *s, count);
    if (options_bytes) {
        MESSAGE(void (*)(id, SEL, const void *, metal_index, metal_index))(
            encoder, SELECTOR("setBytes:length:atIndex:"), options, options_bytes, count + 1);
    }
    const metal_index width = MESSAGE(metal_index (*)(id, SEL))(
        pipeline, SELECTOR("threadExecutionWidth"));
    const char *dispatch = work_is_groups ? "dispatchThreadgroups:threadsPerThreadgroup:"
                                          : "dispatchThreads:threadsPerThreadgroup:";
    MESSAGE(void (*)(id, SEL, metal_size, metal_size))(
        encoder, SELECTOR(dispatch), (metal_size){work, 1, 1}, (metal_size){width, 1, 1});
    invoke(encoder, "endEncoding");
    return 0;
}

static int submit_kernel(void *handle, id pipeline, const plait_w4a16_shape *s,
                         const id *buffers, metal_index count, const void *options,
                         size_t options_bytes, metal_index groups, void **completion) {
    id pool = autorelease_pool();
    id command = object((id)handle, "commandBuffer");
    const metal_index work = groups ? groups : (metal_index)s->m * s->n;
    const int status = encode_kernel(command, pipeline, s, buffers, count, options, options_bytes,
                                     work, groups != 0);
    if (!status) {
        invoke(command, "commit");
        *completion = object(command, "retain");
    }
    invoke(pool, "drain");
    return status;
}

static int device_launch(void *handle, const plait_w4a16_shape *s,
                         const void *a, const void *w, const void *scale,
                         const void *zero, const void *bias, void *out, void **completion) {
    if (!plait_w4a16_valid_shape(s) || !a || !w || !scale || !zero || !out) return 1;
    // The shader skips the bias binding when the optional input is omitted.
    const id buffers[] = {(id)a, (id)w, (id)scale, (id)zero, (id)(bias ? bias : scale), (id)out};
    const uint32_t has_bias = bias != NULL;
    return submit_kernel(handle, metal.pipeline, s, buffers, 6, &has_bias, sizeof has_bias, 0, completion);
}

int plait_w4_quantized_launch_v1(
    void *handle, const plait_w4a16_shape *s, uint32_t bits, const void *a,
    const void *activation_scale, const void *w, const void *weight_scale, const void *zero,
    const void *bias, void *out, void **completion) {
    if (!plait_w4_quantized_valid_shape(s, bits) || !a || !activation_scale ||
        !w || !weight_scale || !zero || !out) return 1;
    if (!metal.integer_pipeline) return 2;
    const id buffers[] = {
        (id)a, (id)activation_scale, (id)w, (id)weight_scale, (id)zero,
        (id)(bias ? bias : weight_scale), (id)out,
    };
    const uint32_t options[] = {bias != NULL, bits};
    // Eight adjacent lanes reduce one dot product, four K columns per lane.
    // Aligned groups keep both packed loads and the per-group SIMD reduction in bounds.
    const metal_index width = metal.integer_simd_width;
    // Limit the cooperative launch to shapes whose measured latency benefits
    // from the extra threads. Large batches keep the packed per-output kernel.
    const uint32_t max_rows = bits == 4 ? 16 : 64;
    if (width && width % 8 == 0 && s->k >= 512 && s->k % 32 == 0 &&
        s->group_size % 32 == 0 && s->m <= max_rows) {
        const metal_index outputs = (metal_index)s->m * s->n;
        const metal_index per_group = width / 8;
        const metal_index groups = (outputs + per_group - 1) / per_group;
        return submit_kernel(handle, metal.integer_simd_pipeline, s, buffers, 7, options,
                             sizeof options, groups, completion);
    }
    return submit_kernel(handle, metal.integer_pipeline, s, buffers, 7, options,
                         sizeof options, 0, completion);
}

int plait_w4_quantize_f16_v1(
    void *handle, uint32_t rows, uint32_t columns, uint32_t group_size, uint32_t bits,
    const void *input, void *codes, void *scales, void *invalid, void **completion) {
    const plait_w4a16_shape s = {rows, 1, columns, group_size, 16};
    if (!plait_w4_quantized_valid_shape(&s, bits) || (uint64_t)rows * columns > UINT32_MAX ||
        !handle || !input || !codes || !scales || !invalid || !completion) return 1;
    if (!metal.quantize_pipeline || !metal.pack_pipeline) return 2;
    id pool = autorelease_pool();
    id command = object((id)handle, "commandBuffer");
    id blit = object(command, "blitCommandEncoder");
    int status = 3;
    if (!blit) goto done;
    MESSAGE(void (*)(id, SEL, id, metal_range, uint8_t))(
        blit, SELECTOR("fillBuffer:range:value:"), (id)invalid, (metal_range){0, sizeof(uint32_t)}, 0);
    invoke(blit, "endEncoding");

    const id buffers[] = {(id)input, (id)codes, (id)scales, (id)invalid};
    const uint32_t groups = plait_ceil_div(columns, group_size);
    status = encode_kernel(command, metal.quantize_pipeline, &s, buffers, 4, &bits, sizeof bits,
                           (metal_index)rows * groups, true);
    if (!status && bits == 4 && group_size % 2 != 0) {
        const uint32_t stride = plait_ceil_div(columns, 2);
        status = encode_kernel(command, metal.pack_pipeline, &s, buffers, 3, NULL, 0,
                               (metal_index)rows * stride, false);
    }
    if (!status) {
        invoke(command, "commit");
        *completion = object(command, "retain");
    }
done:
    invoke(pool, "drain");
    return status;
}

static const metal_index command_completed = 4;  /* MTLCommandBufferStatusCompleted */

static metal_index command_status(id command) {
    return MESSAGE(metal_index (*)(id, SEL))(command, SELECTOR("status"));
}

static int device_wait(void *handle, void *completion) {
    (void)handle;
    id pool = autorelease_pool();
    id command = (id)completion;
    invoke(command, "waitUntilCompleted");
    const bool succeeded = command_status(command) == command_completed;
    if (!succeeded) show_error(object(command, "error"));
    invoke(pool, "drain");
    return succeeded ? 0 : 4;
}

static void device_completion_free(void *handle, void *completion) {
    (void)handle;
    invoke((id)completion, "release");
}

static const plait_w4a16_device_api api = {
    device_open, device_close, device_buffer_new, device_buffer_free,
    device_buffer_read, device_launch, device_wait, device_completion_free,
};

const plait_w4a16_device_api *plait_w4a16_device_api_v1(void) {
    return &api;
}

int plait_w4_completion_query_v1(void *handle, void *completion, uint32_t *ready) {
    if (!handle || !completion || !ready) return 1;
    const metal_index status = command_status((id)completion);
    if (status > command_completed) return 4;
    *ready = status == command_completed;
    return 0;
}

int plait_w4a16_run_host(const plait_w4a16_shape *s, int ordinal,
                         const uint16_t *a, const uint8_t *w, const uint16_t *scale,
                         const uint8_t *zero, const uint16_t *bias, void *out) {
    if (!plait_w4a16_valid_shape(s) || !a || !w || !scale || !zero || !out) return 1;
    if (ordinal != 0 || pthread_once(&initialized, initialize) || !metal.pipeline) return 2;
    id pool = autorelease_pool();
    const size_t groups = (size_t)s->n * plait_ceil_div(s->k, s->group_size);
    const size_t output_bytes = (size_t)s->m * s->n * (s->output_bits / 8);
    const void *host[] = {a, w, scale, zero, bias, NULL};
    const size_t sizes[] = {
        (size_t)s->m * s->k * 2, (size_t)s->n * plait_ceil_div(s->k, 2),
        groups * 2, groups, (size_t)s->n * 2, output_bytes,
    };
    void *buffers[6] = {0};
    void *event = NULL;
    int result = 0;
    for (size_t i = 0; i < 6; ++i) {
        if (i == 4 && !bias) continue;
        result = device_buffer_new(metal.queue, host[i], sizes[i], &buffers[i]);
        if (result) goto release;
    }
    result = device_launch(metal.queue, s, buffers[0], buffers[1], buffers[2],
                           buffers[3], buffers[4], buffers[5], &event);
    if (!result) result = device_wait(metal.queue, event);
    if (!result) result = device_buffer_read(metal.queue, buffers[5], out, output_bytes);
release:
    device_completion_free(metal.queue, event);
    for (size_t i = 0; i < 6; ++i) device_buffer_free(metal.queue, buffers[i]);
    invoke(pool, "drain");
    return result;
}
