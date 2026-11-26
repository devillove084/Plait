#include <cuda_runtime.h>
#include <stdint.h>

int marlin_cuda(const void *activation, const void *packed_weight, void *output,
                void *scale, int m, int n, int k, void *workspace, int group_size,
                int device, cudaStream_t stream, int thread_k, int thread_n,
                int multiprocessors, int max_parallel);

extern "C" int plait_marlin_launch(
    const uint16_t *activation, const uint32_t *packed_weight, uint16_t *scale,
    uint16_t *output, int32_t *workspace, int m, int n, int k, int group_size,
    int device, void *stream) {
    const int status = marlin_cuda(
        activation, packed_weight, output, scale, m, n, k, workspace,
        group_size, device, static_cast<cudaStream_t>(stream), -1, -1, -1, 16);
    return status ? status : static_cast<int>(cudaGetLastError());
}

#if defined(PLAIT_W4_MARLIN_SHARDS)
int marlin_cuda_f32(const void *activation, const void *packed_weight, void *output,
                    void *scale, int m, int n, int k, void *workspace, int group_size,
                    int device, cudaStream_t stream);

extern "C" int plait_marlin_f32_launch(
    const uint16_t *activation, const uint32_t *packed_weight, uint16_t *scale,
    float *output, int32_t *workspace, int m, int n, int k, int group_size,
    int device, void *stream) {
    const int status = marlin_cuda_f32(
        activation, packed_weight, output, scale, m, n, k, workspace,
        group_size, device, static_cast<cudaStream_t>(stream));
    return status ? status : static_cast<int>(cudaGetLastError());
}
#endif
