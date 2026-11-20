#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at line %d\n",                       \
                   cudaGetErrorString(status), __LINE__);                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != ncclSuccess) {                                               \
      std::fprintf(stderr, "NCCL error %s at line %d\n",                       \
                   ncclGetErrorString(status), __LINE__);                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static void send_all(int fd, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  while (size > 0) {
    ssize_t n = ::send(fd, p, size, 0);
    if (n <= 0) { std::perror("send"); std::exit(1); }
    p += n; size -= static_cast<size_t>(n);
  }
}

static void recv_all(int fd, void* data, size_t size) {
  char* p = static_cast<char*>(data);
  while (size > 0) {
    ssize_t n = ::recv(fd, p, size, 0);
    if (n <= 0) { std::perror("recv"); std::exit(1); }
    p += n; size -= static_cast<size_t>(n);
  }
}

static ncclUniqueId exchange_id(int rank, int world, const char* root_ip,
                                int port) {
  ncclUniqueId id{};
  if (rank == 0) {
    int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { std::perror("socket"); std::exit(1); }
    int yes = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
      std::perror("bind"); std::exit(1);
    }
    if (::listen(server, world) < 0) { std::perror("listen"); std::exit(1); }
    NCCL_CHECK(ncclGetUniqueId(&id));
    for (int peer = 1; peer < world; ++peer) {
      int fd = ::accept(server, nullptr, nullptr);
      if (fd < 0) { std::perror("accept"); std::exit(1); }
      send_all(fd, &id, sizeof(id));
      ::close(fd);
    }
    ::close(server);
  } else {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { std::perror("socket"); std::exit(1); }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (::inet_pton(AF_INET, root_ip, &address.sin_addr) != 1) {
      std::fprintf(stderr, "bad root ip %s\n", root_ip); std::exit(1);
    }
    for (int attempt = 0; attempt < 120; ++attempt) {
      if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) break;
      ::close(fd); ::usleep(250000);
      fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (attempt == 119) { std::perror("connect"); std::exit(1); }
    }
    recv_all(fd, &id, sizeof(id));
    ::close(fd);
  }
  return id;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr,
                 "usage: %s rank world gpu bytes iters root_ip port\n", argv[0]);
    return 1;
  }
  int rank = std::atoi(argv[1]);
  int world = std::atoi(argv[2]);
  int gpu = std::atoi(argv[3]);
  size_t bytes = std::strtoull(argv[4], nullptr, 10);
  int iters = std::atoi(argv[5]);
  const char* root_ip = argv[6];
  int port = std::atoi(argv[7]);
  size_t count = bytes / sizeof(float);

  CHECK(cudaSetDevice(gpu));
  ncclUniqueId id = exchange_id(rank, world, root_ip, port);
  ncclComm_t comm{};
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  float* device = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  std::vector<float> host(count);
  for (size_t i = 0; i < count; ++i) host[i] = static_cast<float>(rank + 1);
  CHECK(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
  NCCL_CHECK(ncclAllReduce(device, device, count, ncclFloat, ncclSum, comm,
                           nullptr));
  CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    NCCL_CHECK(ncclAllReduce(device, device, count, ncclFloat, ncclSum, comm,
                             nullptr));
  }
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(end - start).count() / iters;

  CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
  double expected = static_cast<double>(world * (world + 1)) / 2.0;
  int errors = 0;
  for (size_t i = 0; i < count; ++i) {
    if (std::abs(host[i] - expected) > 1e-2) ++errors;
  }
  double algbw = static_cast<double>(bytes) / seconds / 1e9;
  double busbw = algbw * 2.0 * (world - 1) / world;
  std::printf("rank=%d world=%d gpu=%d bytes=%zu ms=%.3f algbw=%.3f busbw=%.3f errors=%d\n",
              rank, world, gpu, bytes, seconds * 1e3, algbw, busbw, errors);
  NCCL_CHECK(ncclCommDestroy(comm));
  CHECK(cudaFree(device));
  return errors == 0 ? 0 : 2;
}
