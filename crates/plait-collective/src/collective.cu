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
    int mode,
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
    if (mode != 1) {
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
    }
    if (mode != 2) {
    for (int step = 0; step < ranks - 1; ++step) {
        int send_chunk = (mode == 1)
            ? (rank + ranks - step) % ranks
            : (rank + ranks - step + 1) % ranks;
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
}

extern "C" __global__ void plait_alltoall(
    float* output,
    unsigned long long* peers,
    int rank,
    int ranks,
    int chunk)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int source = 0; source < ranks; ++source) {
        const float* src = reinterpret_cast<const float*>(peers[source]);
        for (int i = tid; i < chunk; i += stride) {
            output[source * chunk + i] = src[rank * chunk + i];
        }
    }
}

extern "C" __global__ void plait_broadcast_chain(
    float* data,
    unsigned int* flags,
    unsigned long long next_data,
    unsigned long long next_flags,
    int rank,
    int ranks,
    int root,
    int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float* next = reinterpret_cast<float*>(next_data);
    unsigned int* next_flags_ptr = reinterpret_cast<unsigned int*>(next_flags);
    for (int step = 0; step < ranks - 1; ++step) {
        int source = (root + step) % ranks;
        int destination = (source + 1) % ranks;
        if (rank == source) {
            for (int i = tid; i < count; i += stride) next[i] = data[i];
            __threadfence_system();
            if (tid == 0) next_flags_ptr[step] = (unsigned int)(step + 1);
        }
        if (rank == destination) {
            if (tid == 0) {
                while (*(volatile unsigned int*)(flags + step) != (unsigned int)(step + 1)) {
                }
            }
            __syncthreads();
            __threadfence_system();
        }
        __syncthreads();
    }
}

extern "C" __global__ void plait_broadcast_fanout(
    const float* source,
    unsigned long long* peers,
    int ranks,
    int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int peer = 0; peer < ranks; ++peer) {
        float* destination = reinterpret_cast<float*>(peers[peer]);
        for (int i = tid; i < count; i += stride) {
            destination[i] = source[i];
        }
    }
}

extern "C" __global__ void plait_alltoall_push(
    const float* input,
    unsigned long long* output_peers,
    int rank,
    int ranks,
    int chunk)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int destination = 0; destination < ranks; ++destination) {
        float* dst = reinterpret_cast<float*>(output_peers[destination]);
        for (int i = tid; i < chunk; i += stride) {
            dst[rank * chunk + i] = input[destination * chunk + i];
        }
    }
}

extern "C" __global__ void plait_scatter_root(
    const float* source,
    unsigned long long* peers,
    int ranks,
    int width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int peer = 0; peer < ranks; ++peer) {
        float* destination = reinterpret_cast<float*>(peers[peer]);
        for (int i = tid; i < width; i += stride) {
            destination[peer * width + i] = source[peer * width + i];
        }
    }
}

extern "C" __global__ void plait_scatter_pull(
    const float* root,
    float* local,
    int rank,
    int width)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < width; i += stride) {
        local[rank * width + i] = root[rank * width + i];
    }
}

extern "C" __global__ void plait_ll128_allreduce(
    float* data,
    float* recv,
    unsigned int* flags,
    unsigned long long next_data,
    unsigned long long next_recv,
    unsigned long long next_flags,
    int rank,
    int ranks,
    int width,
    int lines,
    int line_begin,
    int line_end)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* next_data_ptr = reinterpret_cast<float*>(next_data);
    float* next_recv_ptr = reinterpret_cast<float*>(next_recv);
    unsigned int* next_flags_ptr = reinterpret_cast<unsigned int*>(next_flags);
    int prev = (rank + ranks - 1) % ranks;
    for (int step = 0; step < ranks - 1; ++step) {
        int send_chunk = (rank + ranks - step) % ranks;
        int recv_chunk = (rank + ranks - step - 1) % ranks;
        for (int line = line_begin; line < line_end; ++line) {
            int start = line * 32;
            int end = min(width, start + 32);
            for (int i = start + tid; i < end; i += blockDim.x) {
                next_recv_ptr[step * width + i] = data[send_chunk * width + i];
            }
            __syncthreads();
            __threadfence_system();
            if (tid == 0) {
                next_flags_ptr[step * lines + line] = (unsigned int)(rank + 1);
                while (*(volatile unsigned int*)(flags + step * lines + line) !=
                       (unsigned int)(prev + 1)) {
                }
            }
            __syncthreads();
            __threadfence_system();
            for (int i = start + tid; i < end; i += blockDim.x) {
                data[recv_chunk * width + i] += recv[step * width + i];
            }
            __syncthreads();
        }
    }
    for (int step = 0; step < ranks - 1; ++step) {
        int send_chunk = (rank + ranks - step + 1) % ranks;
        for (int line = line_begin; line < line_end; ++line) {
            int start = line * 32;
            int end = min(width, start + 32);
            for (int i = start + tid; i < end; i += blockDim.x) {
                next_data_ptr[send_chunk * width + i] = data[send_chunk * width + i];
            }
            __syncthreads();
            __threadfence_system();
            if (tid == 0) {
                next_flags_ptr[(ranks - 1) * lines + step * lines + line] =
                    (unsigned int)(rank + 1);
                while (*(volatile unsigned int*)(flags + (ranks - 1) * lines +
                                                 step * lines + line) !=
                       (unsigned int)(prev + 1)) {
                }
            }
            __syncthreads();
            __threadfence_system();
        }
    }
}
