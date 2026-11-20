extern "C" __global__ void plait_axpy(
    const float* x,
    const float* y,
    float* out,
    float alpha,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}
