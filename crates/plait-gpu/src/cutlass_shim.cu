// CUTLASS-backed SGEMM with a tiny C ABI for Rust FFI.
// A and B are row-major; C = alpha * A * B + beta * C.
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cstdlib>

using RowMajor = cutlass::layout::RowMajor;
using CutlassSgemm = cutlass::gemm::device::Gemm<
    float, RowMajor,
    float, RowMajor,
    float, RowMajor,
    float>;

extern "C" int plait_cutlass_sgemm(
    const float* host_a,
    const float* host_b,
    float* host_c,
    int m,
    int n,
    int k,
    float alpha,
    float beta)
{
    if (m <= 0 || n <= 0 || k <= 0 || host_a == nullptr || host_b == nullptr || host_c == nullptr) {
        return 1;
    }
    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
    const size_t b_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    if (cudaMalloc(&a, a_bytes) != cudaSuccess ||
        cudaMalloc(&b, b_bytes) != cudaSuccess ||
        cudaMalloc(&c, c_bytes) != cudaSuccess) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        return 2;
    }
    if (cudaMemcpy(a, host_a, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(b, host_b, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(c, host_c, c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        return 3;
    }

    CutlassSgemm gemm;
    CutlassSgemm::Arguments args(
        {m, n, k},
        {a, k},
        {b, n},
        {c, n},
        {c, n},
        {alpha, beta});
    const cutlass::Status status = gemm(args);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        return 4;
    }
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_c, c, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        return 5;
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}

extern "C" int plait_cutlass_sgemm_device(
    const void* device_a,
    const void* device_b,
    void* device_c,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    void* stream_ptr)
{
    if (m <= 0 || n <= 0 || k <= 0 || device_a == nullptr ||
        device_b == nullptr || device_c == nullptr) {
        return 1;
    }
    const float* a = static_cast<const float*>(device_a);
    const float* b = static_cast<const float*>(device_b);
    float* c = static_cast<float*>(device_c);
    CutlassSgemm gemm;
    CutlassSgemm::Arguments args(
        {m, n, k},
        {a, k},
        {b, n},
        {c, n},
        {c, n},
        {alpha, beta});
    void* workspace = nullptr;
    const size_t workspace_bytes = CutlassSgemm::get_workspace_size(args);
    if (workspace_bytes != 0 && cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
        return 2;
    }
    const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const cutlass::Status status = gemm(args, workspace, stream);
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    return status == cutlass::Status::kSuccess ? 0 : 3;
}
