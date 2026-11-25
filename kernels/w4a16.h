#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t group_size;
    uint32_t output_bits;
} plait_w4a16_shape;

/* Row-major A[M,K] uses IEEE binary16. W[N,ceil(K/2)] packs
 * unsigned INT4 with even K in the low nibble. Per-channel, per-group scale
 * (binary16) and zero point (uint8 in [0,15]) have shape [N,ceil(K/group_size)].
 * The effective weight is (code - zero) * scale. Optional bias[N] uses
 * binary16. Accumulation is in binary32. Output[M,N] uses binary16 or
 * binary32 as selected by output_bits. Use binary32 for K-shard partials.
 *
 * All pointers must be valid for the indicated full matrix on the same
 * device. Output must not alias the inputs. Launch uses the current device
 * and the runtime's caller-owned stream. Zero means submission succeeded.
 */
int plait_w4a16_launch(const plait_w4a16_shape *shape,
                       const uint16_t *a, const uint8_t *w, const uint16_t *scale, const uint8_t *zero,
                       const uint16_t *bias, void *out, void *stream);

/* Synchronous staging entry for conformance tests and host callers. */
int plait_w4a16_run_host(const plait_w4a16_shape *shape, int ordinal,
                         const uint16_t *a, const uint8_t *w, const uint16_t *scale, const uint8_t *zero,
                         const uint16_t *bias, void *out);

/* Internal device-resident ABI. A session owns one in-order execution queue.
 * Buffers and completions belong to that session. buffer_new consumes the
 * host bytes before returning, or allocates uninitialized storage when
 * source is null. launch queues work and returns an owned completion.
 * buffer_read requires completed input. All destroy operations are safe
 * while earlier commands are in flight and may wait if needed.
 */
typedef struct {
    int (*open)(int ordinal, void **session);
    void (*close)(void *session);
    int (*buffer_new)(void *session, const void *source, size_t bytes, void **buffer);
    void (*buffer_free)(void *session, void *buffer);
    int (*buffer_read)(void *session, const void *buffer, void *destination, size_t bytes);
    int (*launch)(void *session, const plait_w4a16_shape *shape,
                  const void *activation, const void *weight, const void *scale,
                  const void *zero, const void *bias, void *output, void **completion);
    int (*wait)(void *session, void *completion);
    void (*completion_free)(void *session, void *completion);
} plait_w4a16_device_api;

const plait_w4a16_device_api *plait_w4a16_device_api_v1(void);

/* Optional completion query for a handle returned by submission on this session.
 * Returns 0 with ready set to 0 (pending) or 1 (finished). A nonzero result is
 * the backend error status. The query does not wait, copy results, or release
 * the completion. Read a quantizer's invalid flag separately after completion.
 */
int plait_w4_completion_query_v1(void *session, void *completion, uint32_t *ready);

/* Optional integer activation GEMM on an existing session queue. Activations
 * are signed INT4 (low nibble first) or INT8. Each row and K group has an FP32
 * activation scale. The dot is accumulated in INT32 for each group, then
 * dequantized using its activation and weight scales before FP32 summation.
 */
int plait_w4_quantized_launch_v1(
    void *session, const plait_w4a16_shape *shape, uint32_t activation_bits,
    const void *activation, const void *activation_scale, const void *weight,
    const void *weight_scale, const void *zero, const void *bias, void *output,
    void **completion);

/* Optional quantizer on the same in-order queue. Outputs use the same layout
 * as plait_w4_quantized_launch_v1. invalid points to one device uint32_t, set
 * to 1 if any input is non-finite and 0 otherwise, observable at completion.
 */
int plait_w4_quantize_f16_v1(
    void *session, uint32_t rows, uint32_t columns, uint32_t group_size,
    uint32_t activation_bits, const void *input, void *codes, void *scales,
    void *invalid, void **completion);

/* Optional CUDA-compatible extension. Buffers and completions use the same session ABI.
 * probe returns -1 for an unsupported device, or a runtime status otherwise.
 * launch accepts FP16 without bias and Marlin-packed weights and scales.
 * Workspace is at least (n / 128) * 16 int32_t values and is cleared by launch.
 */
typedef struct {
    int (*probe)(void *session);
    int (*launch)(void *session, const plait_w4a16_shape *shape,
                  const void *activation, const void *weight, const void *scale,
                  void *workspace, void *output, void **completion);
} plait_w4a16_marlin_api;

const plait_w4a16_marlin_api *plait_w4a16_marlin_api_v1(void);

/* Optional AMD CDNA3 Marlin port. The same launch supports FP16 and FP32 result
 * types. Each 16-by-16 weight tile has one word of four INT4 codes per wave lane.
 * Tiles are ordered by output tile, then input tile. Group-major parameters
 * place the FP16 scale in the low 16 bits and the unsigned zero point above it.
 * The workspace query returns zero when the wave tiles need no split-K merge.
 */
typedef struct {
    plait_w4a16_marlin_api kernel;
    size_t (*workspace)(uint32_t rows, uint32_t columns, uint32_t input_columns);
} plait_w4a16_marlin_mfma_api;

const plait_w4a16_marlin_mfma_api *plait_w4a16_marlin_mfma_api_v1(void);

/* Optional FP32 split-K accumulation and final output with Marlin-packed weights.
 * Available when Marlin is built with PLAIT_W4_MARLIN_SHARDS.
 * If also exported, the workspace query gives the bytes needed for the shape.
 * Otherwise the FP16 ABI's lock workspace size applies.
 */
size_t plait_w4a16_marlin_f32_workspace_v1(uint32_t rows, uint32_t columns);

int plait_w4a16_marlin_f32_launch_v1(
    void *session, const plait_w4a16_shape *shape, const void *activation,
    const void *weight, const void *scale, void *workspace, void *output, void **completion);

/* Optional CUDA-compatible Humming extension on the same queue. probe returns -1
 * when the device or weight shape lacks a compiled kernel. repack transforms
 * canonical weights into an equally sized device buffer. launch accepts
 * the prepared weights, group-major FP16 scales and packed INT4 zero points.
 */
typedef struct {
    int (*probe)(void *session, uint32_t n, uint32_t k, uint32_t group_size);
    int (*repack)(void *session, const void *canonical, void *packed);
    int (*launch)(void *session, const plait_w4a16_shape *shape,
                  const void *activation, const void *weight, const void *scale,
                  const void *zero, void *output, void **completion);
} plait_w4a16_humming_api;

const plait_w4a16_humming_api *plait_w4a16_humming_api_v1(void);

/* Optional FP32 output on the same prepared Humming buffers and queue.
 * Available when Humming is built with PLAIT_W4_HUMMING_SHARDS.
 */
int plait_w4a16_humming_f32_launch_v1(
    void *session, const plait_w4a16_shape *shape, const void *activation,
    const void *weight, const void *scale, const void *zero, void *output, void **completion);

#ifdef __cplusplus
}
#endif
