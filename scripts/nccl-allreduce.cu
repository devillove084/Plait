// Minimal NCCL all-reduce correctness/benchmark harness.
//
// This mirrors the parts of NVIDIA/nccl-tests that matter for the Plait
// comparison: deterministic input, optional CPU reference check, warmup
// iterations, timed iterations, algorithm bandwidth, and bus bandwidth.
// It is intentionally one process with one NCCL communicator per visible GPU,
// so it can run in this container without MPI.

#include <nccl.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(error), __FILE__, __LINE__);     \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

#define NCCL_CHECK(call)                                                     \
    do {                                                                     \
        ncclResult_t result = (call);                                        \
        if (result != ncclSuccess) {                                         \
            std::fprintf(stderr, "NCCL error %s at %s:%d\n",                 \
                         ncclGetErrorString(result), __FILE__, __LINE__);    \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

struct Options {
    int gpus = 0;
    int iters = 20;
    int warmup = 5;
    int m = 8;
    int n = 8;
    int k = 6;
    int logical = 12;
    bool check = true;
    bool allreduce = false;
    size_t min_bytes = 4096;
    size_t max_bytes = 64ull << 20;
    size_t step_factor = 4;
};

static Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "--gpus") options.gpus = std::atoi(next());
        else if (arg == "--iters") options.iters = std::atoi(next());
        else if (arg == "--warmup") options.warmup = std::atoi(next());
        else if (arg == "--m") options.m = std::atoi(next());
        else if (arg == "--n") options.n = std::atoi(next());
        else if (arg == "--k") options.k = std::atoi(next());
        else if (arg == "--logical") options.logical = std::atoi(next());
        else if (arg == "--min-bytes") options.min_bytes = std::strtoull(next(), nullptr, 0);
        else if (arg == "--max-bytes") options.max_bytes = std::strtoull(next(), nullptr, 0);
        else if (arg == "--step") options.step_factor = std::strtoull(next(), nullptr, 0);
        else if (arg == "--no-check") options.check = false;
        else if (arg == "--allreduce") options.allreduce = true;
        else if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: %s [--gpus N] [--iters N] [--warmup N] [--m M --n N --k K]\n"
                "          [--logical L] [--allreduce --min-bytes B --max-bytes B --step F]\n"
                "          [--no-check]\n",
                argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            std::exit(2);
        }
    }
    return options;
}

static float a_value(int index) {
    return static_cast<float>((index * 7 + 3) % 17 - 8);
}

static float b_value(int index) {
    return static_cast<float>((index * 11 + 5) % 13 - 6);
}

// Same DP2 x TP3 x EP2 tiling as plait-node's matrix runner.
static void matrix_partial(int m, int n, int k, int contribution,
                           std::vector<float>& out) {
    const int dp = contribution % 2;
    const int tp = (contribution / 2) % 3;
    const int ep = (contribution / 6) % 2;
    const int row_start = dp * m / 2;
    const int row_end = (dp + 1) * m / 2;
    const int col_start = ep * n / 2;
    const int col_end = (ep + 1) * n / 2;
    const int inner_start = tp * k / 3;
    const int inner_end = (tp + 1) * k / 3;
    out.assign(m * n, 0.0f);
    for (int row = row_start; row < row_end; ++row) {
        for (int col = col_start; col < col_end; ++col) {
            double sum = 0.0;
            for (int inner = inner_start; inner < inner_end; ++inner) {
                sum += static_cast<double>(a_value(row * k + inner)) *
                       static_cast<double>(b_value(inner * n + col));
            }
            out[row * n + col] = static_cast<float>(sum);
        }
    }
}

static double checksum(const std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) total += value;
    return total;
}

static double checksum(const std::vector<float>& values) {
    double total = 0.0;
    for (float value : values) total += value;
    return total;
}

static int count_errors(const std::vector<float>& actual,
                        const std::vector<double>& expected) {
    int errors = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double tolerance = 2e-4 * std::max(1.0, std::fabs(expected[i]));
        if (std::fabs(static_cast<double>(actual[i]) - expected[i]) > tolerance) {
            ++errors;
        }
    }
    return errors;
}

struct Comm {
    std::vector<int> devices;
    std::vector<ncclComm_t> comms;
    std::vector<cudaStream_t> streams;
};

static Comm init_comms(int requested) {
    int visible = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visible));
    int gpus = requested > 0 ? requested : visible;
    if (gpus < 1 || gpus > visible) {
        std::fprintf(stderr, "requested %d GPUs, %d visible\n", gpus, visible);
        std::exit(2);
    }
    Comm state;
    state.devices.resize(gpus);
    state.comms.resize(gpus);
    state.streams.resize(gpus);
    for (int i = 0; i < gpus; ++i) state.devices[i] = i;
    NCCL_CHECK(ncclCommInitAll(state.comms.data(), gpus, state.devices.data()));
    for (int i = 0; i < gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&state.streams[i]));
    }
    return state;
}

static void destroy_comms(Comm& state) {
    for (ncclComm_t comm : state.comms) {
        ncclCommDestroy(comm);
    }
    for (cudaStream_t stream : state.streams) {
        cudaStreamDestroy(stream);
    }
}

static void allreduce(Comm& state, const std::vector<float*>& send,
                      std::vector<float*>& recv, size_t count) {
    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < state.comms.size(); ++i) {
        NCCL_CHECK(ncclAllReduce(send[i], recv[i], count, ncclFloat, ncclSum,
                                 state.comms[i], state.streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int i = 0; i < static_cast<int>(state.comms.size()); ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(state.streams[i]));
    }
}

static double timed_allreduce(Comm& state, const std::vector<float*>& send,
                              std::vector<float*>& recv, size_t count,
                              const Options& options) {
    for (int i = 0; i < options.warmup; ++i) allreduce(state, send, recv, count);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iters; ++i) allreduce(state, send, recv, count);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count() /
           static_cast<double>(options.iters);
}

static void print_bandwidth(size_t bytes, double seconds, int gpus,
                            int errors) {
    double algbw = bytes / seconds / 1e9;
    double busbw = algbw * 2.0 * (gpus - 1) / gpus;
    std::printf("size=%zu time_ms=%.6f algbw_GB/s=%.3f busbw_GB/s=%.3f errors=%d\n",
                bytes, seconds * 1e3, algbw, busbw, errors);
}

static int run_matrix(Comm& state, const Options& options) {
    const int gpus = static_cast<int>(state.comms.size());
    const size_t count = static_cast<size_t>(options.m) * options.n;
    std::vector<std::vector<float>> host_send(gpus, std::vector<float>(count, 0.0f));
    std::vector<double> reference(count, 0.0);
    for (int contribution = 0; contribution < options.logical; ++contribution) {
        std::vector<float> partial;
        matrix_partial(options.m, options.n, options.k, contribution, partial);
        int rank = contribution % gpus;
        for (size_t i = 0; i < count; ++i) {
            host_send[rank][i] += partial[i];
            reference[i] += static_cast<double>(partial[i]);
        }
    }

    std::vector<float*> send(gpus);
    std::vector<float*> recv(gpus);
    std::vector<std::vector<float>> host_recv(gpus, std::vector<float>(count));
    for (int i = 0; i < gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&send[i], count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recv[i], count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(send[i], host_send[i].data(), count * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    double seconds = timed_allreduce(state, send, recv, count, options);
    for (int i = 0; i < gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemcpy(host_recv[i].data(), recv[i], count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(send[i]));
        CUDA_CHECK(cudaFree(recv[i]));
    }

    int errors = 0;
    if (options.check) {
        for (int i = 0; i < gpus; ++i) {
            errors = std::max(errors, count_errors(host_recv[i], reference));
        }
    }
    double expected_sum = checksum(reference);
    double actual_sum = checksum(host_recv[0]);
    std::printf("matrix=%dx%dx%d logical=%d gpus=%d checksum_expected=%.9f "
                "checksum_nccl=%.9f max_errors=%d\n",
                options.m, options.n, options.k, options.logical, gpus,
                expected_sum, actual_sum, errors);
    print_bandwidth(count * sizeof(float), seconds, gpus, errors);
    return errors == 0 ? 0 : 1;
}

static int run_allreduce(Comm& state, const Options& options) {
    const int gpus = static_cast<int>(state.comms.size());
    size_t bytes = options.min_bytes;
    while (bytes <= options.max_bytes) {
        size_t count = bytes / sizeof(float);
        if (count == 0) count = 1;
        std::vector<std::vector<float>> host_send(gpus);
        std::vector<std::vector<float>> host_recv(gpus, std::vector<float>(count));
        for (int rank = 0; rank < gpus; ++rank) {
            host_send[rank].resize(count);
            for (size_t i = 0; i < count; ++i) {
                host_send[rank][i] = static_cast<float>(rank + 1) +
                                     0.001f * static_cast<float>(i % 17);
            }
        }
        std::vector<float*> send(gpus);
        std::vector<float*> recv(gpus);
        for (int i = 0; i < gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMalloc(&send[i], count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&recv[i], count * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(send[i], host_send[i].data(), count * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        double seconds = timed_allreduce(state, send, recv, count, options);
        for (int i = 0; i < gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemcpy(host_recv[i].data(), recv[i], count * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(send[i]));
            CUDA_CHECK(cudaFree(recv[i]));
        }
        int errors = 0;
        if (options.check) {
            std::vector<double> expected(count);
            for (size_t i = 0; i < count; ++i) {
                double sum = 0.0;
                for (int rank = 0; rank < gpus; ++rank) sum += host_send[rank][i];
                expected[i] = sum;
            }
            for (int rank = 0; rank < gpus; ++rank) {
                errors = std::max(errors, count_errors(host_recv[rank], expected));
            }
        }
        print_bandwidth(count * sizeof(float), seconds, gpus, errors);
        if (bytes > options.max_bytes / options.step_factor) break;
        bytes *= options.step_factor;
    }
    return 0;
}

int main(int argc, char** argv) {
    Options options = parse(argc, argv);
    Comm state = init_comms(options.gpus);
    int version = 0;
    NCCL_CHECK(ncclGetVersion(&version));
    std::printf("nccl version=%d gpus=%zu\n", version, state.comms.size());
    int result = options.allreduce ? run_allreduce(state, options)
                                    : run_matrix(state, options);
    destroy_comms(state);
    return result;
}