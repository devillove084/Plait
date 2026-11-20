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

extern "C" __global__ void plait_ring_allreduce(
    float* data,
    float* recv,
    unsigned int* flags,
    unsigned long long next_data,
    unsigned long long next_recv,
    unsigned long long next_flags,
    int rank,
    int ranks,
    int count,
    int width,
    int channels,
    int data_stride,
    int recv_stride,
    int flag_stride)
{
    int channel = blockIdx.x;
    if (channel >= channels) {
        return;
    }
    int tid = threadIdx.x;
    float* next_data_ptr = reinterpret_cast<float*>(next_data) + channel * data_stride;
    float* next_recv_ptr = reinterpret_cast<float*>(next_recv) + channel * recv_stride;
    unsigned int* next_flags_ptr = reinterpret_cast<unsigned int*>(next_flags) + channel * flag_stride;
    data += channel * data_stride;
    recv += channel * recv_stride;
    flags += channel * flag_stride;
    int prev = (rank + ranks - 1) % ranks;
    (void)count;
    bool vector = (width % 4) == 0
        && ((reinterpret_cast<unsigned long long>(data) & 15ull) == 0)
        && ((reinterpret_cast<unsigned long long>(next_recv_ptr) & 15ull) == 0)
        && ((reinterpret_cast<unsigned long long>(recv) & 15ull) == 0);
    for (int step = 0; step < ranks - 1; ++step) {
        int send_chunk = (rank + ranks - step) % ranks;
        int recv_chunk = (rank + ranks - step - 1) % ranks;
        if (vector) {
            float4* dst = reinterpret_cast<float4*>(next_recv_ptr + step * width);
            const float4* src = reinterpret_cast<const float4*>(data + send_chunk * width);
            for (int i = tid; i < width / 4; i += blockDim.x) dst[i] = src[i];
        } else {
            for (int i = tid; i < width; i += blockDim.x) {
                next_recv_ptr[step * width + i] = data[send_chunk * width + i];
            }
        }
        __threadfence_system();
        if (tid == 0) {
            next_flags_ptr[step] = (unsigned int)(rank + 1);
        }
        if (tid == 0) {
            while (*(volatile unsigned int*)(flags + step) != (unsigned int)(prev + 1)) {
            }
        }
        __syncthreads();
        __threadfence_system();
        if (vector) {
            float4* dst = reinterpret_cast<float4*>(data + recv_chunk * width);
            const float4* src = reinterpret_cast<const float4*>(recv + step * width);
            for (int i = tid; i < width / 4; i += blockDim.x) {
                float4 a = dst[i];
                float4 b = src[i];
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                dst[i] = a;
            }
        } else {
            for (int i = tid; i < width; i += blockDim.x) {
                data[recv_chunk * width + i] += recv[step * width + i];
            }
        }
        __syncthreads();
    }
    for (int step = 0; step < ranks - 1; ++step) {
        int send_chunk = (rank + ranks - step + 1) % ranks;
        if (vector) {
            float4* dst = reinterpret_cast<float4*>(next_data_ptr + send_chunk * width);
            const float4* src = reinterpret_cast<const float4*>(data + send_chunk * width);
            for (int i = tid; i < width / 4; i += blockDim.x) dst[i] = src[i];
        } else {
            for (int i = tid; i < width; i += blockDim.x) {
                next_data_ptr[send_chunk * width + i] = data[send_chunk * width + i];
            }
        }
        __threadfence_system();
        if (tid == 0) {
            next_flags_ptr[ranks - 1 + step] = (unsigned int)(rank + 1);
        }
        if (tid == 0) {
            while (*(volatile unsigned int*)(flags + ranks - 1 + step) != (unsigned int)(prev + 1)) {
            }
        }
        __syncthreads();
        __threadfence_system();
    }
}
