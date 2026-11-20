extern "C" __global__ void plait_add_inplace(
    float* dst,
    const float* src,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] += src[i];
    }
}
