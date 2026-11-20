#include <arpa/inet.h>
#include <cuda_runtime.h>
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

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s rank world gpu bytes iters root_ip port\n", argv[0]);
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
  float* device = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  std::vector<float> input(count), received(count), sum(count);
  for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(rank + 1);
  CHECK(cudaMemcpy(device, input.data(), bytes, cudaMemcpyHostToDevice));

  int root_fd = -1;
  std::vector<int> peers;
  std::vector<int> ranks;
  if (rank == 0) {
    int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { std::perror("socket"); return 1; }
    int yes = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
      std::perror("bind"); return 1;
    }
    if (::listen(server, world) < 0) { std::perror("listen"); return 1; }
    for (int i = 1; i < world; ++i) {
      int fd = ::accept(server, nullptr, nullptr);
      if (fd < 0) { std::perror("accept"); return 1; }
      int peer_rank = -1;
      recv_all(fd, &peer_rank, sizeof(peer_rank));
      peers.push_back(fd);
      ranks.push_back(peer_rank);
    }
    ::close(server);
  } else {
    root_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (root_fd < 0) { std::perror("socket"); return 1; }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (::inet_pton(AF_INET, root_ip, &address.sin_addr) != 1) {
      std::fprintf(stderr, "bad root ip %s\n", root_ip); return 1;
    }
    for (int attempt = 0; attempt < 120; ++attempt) {
      if (::connect(root_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) break;
      ::close(root_fd); ::usleep(250000);
      root_fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (attempt == 119) { std::perror("connect"); return 1; }
    }
    send_all(root_fd, &rank, sizeof(rank));
  }

  auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iters; ++iteration) {
    // The GPU buffer is staged through host memory; this is the TCP fallback,
    // not GPUDirect.
    CHECK(cudaMemcpy(received.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (rank == 0) {
      std::fill(sum.begin(), sum.end(), 0.0f);
      for (size_t i = 0; i < count; ++i) sum[i] += input[i];
      for (size_t p = 0; p < peers.size(); ++p) {
        recv_all(peers[p], received.data(), bytes);
        for (size_t i = 0; i < count; ++i) sum[i] += received[i];
      }
      for (int fd : peers) send_all(fd, sum.data(), bytes);
      std::copy(sum.begin(), sum.end(), received.begin());
    } else {
      send_all(root_fd, input.data(), bytes);
      recv_all(root_fd, received.data(), bytes);
    }
    CHECK(cudaMemcpy(device, received.data(), bytes, cudaMemcpyHostToDevice));
  }
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(end - start).count() / iters;

  CHECK(cudaMemcpy(received.data(), device, bytes, cudaMemcpyDeviceToHost));
  double expected = static_cast<double>(world * (world + 1)) / 2.0;
  int errors = 0;
  for (size_t i = 0; i < count; ++i) if (std::abs(received[i] - expected) > 1e-2) ++errors;
  double algbw = static_cast<double>(bytes) / seconds / 1e9;
  double busbw = algbw * 2.0 * (world - 1) / world;
  std::printf("rank=%d world=%d gpu=%d bytes=%zu ms=%.3f algbw=%.3f busbw=%.3f errors=%d\n",
              rank, world, gpu, bytes, seconds * 1e3, algbw, busbw, errors);
  for (int fd : peers) ::close(fd);
  if (rank != 0) ::close(root_fd);
  CHECK(cudaFree(device));
  return errors == 0 ? 0 : 2;
}
