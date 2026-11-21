#include <cuda_bf16.h>
#include <cuda_fp16.h>
extern "C" __global__ void plait_kernel(
    unsigned long long input, unsigned long long output, int count, int dtype,
    unsigned int generation, volatile unsigned int* ready, float* slots,
    unsigned int* done, int rank, int capacity)
{
  const void* in = (const void*)input;
  void* out = (void*)output;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    float value;
    if (dtype == 7) value = ((const float*)in)[i];
    else if (dtype == 9) value = __bfloat162float(((const __nv_bfloat16*)in)[i]);
    else value = __half2float(((const __half*)in)[i]);
    slots[rank * capacity + i] = value;
  }
  __threadfence_system();
  if (blockIdx.x == 0 && threadIdx.x == 0) ready[rank] = generation;
  __threadfence_system();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (ready[1 - rank] != generation) __nanosleep(200);
  }
  __syncthreads();
  __threadfence_system();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    float value = slots[rank * capacity + i] + slots[(1 - rank) * capacity + i];
    if (dtype == 7) ((float*)out)[i] = value;
    else if (dtype == 9) ((__nv_bfloat16*)out)[i] = __float2bfloat16(value);
    else ((__half*)out)[i] = __float2half(value);
  }
  __syncthreads();
  __threadfence_system();
  if (blockIdx.x == 0 && threadIdx.x == 0) done[0] = generation;
}
