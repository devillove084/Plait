#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <chrono>
#include <thread>
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
    ssize_t n = ::send(fd, p, size, MSG_NOSIGNAL);
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
static int connect_to(const char* ip, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  ::inet_pton(AF_INET, ip, &a.sin_addr);
  for (int i = 0; i < 240; ++i) {
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) return fd;
    ::close(fd); ::usleep(250000);
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (i == 239) { std::perror("connect"); std::exit(1); }
  }
  return -1;
}
static int listen_on(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = INADDR_ANY;
  a.sin_port = htons(port);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0) { std::perror("bind"); std::exit(1); }
  if (::listen(fd, 16) < 0) { std::perror("listen"); std::exit(1); }
  return fd;
}

// Full-duplex exchange between the two host leaders. Both sides send and
// receive the same chunk at once instead of serializing a 16 MiB block.
static void exchange_bidirectional(int fd, const float* send_buf, float* recv_buf,
                                   size_t count) {
  size_t total = count * sizeof(float);
  int flags = ::fcntl(fd, F_GETFL, 0);
  ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  const char* send_data = reinterpret_cast<const char*>(send_buf);
  char* recv_data = reinterpret_cast<char*>(recv_buf);
  size_t sent = 0;
  size_t received = 0;
  while (sent < total || received < total) {
    pollfd event{};
    event.fd = fd;
    event.events = 0;
    if (sent < total) event.events |= POLLOUT;
    if (received < total) event.events |= POLLIN;
    if (::poll(&event, 1, -1) < 0) {
      if (errno == EINTR) continue;
      std::perror("poll"); std::exit(1);
    }
    if ((event.revents & POLLOUT) && sent < total) {
      ssize_t n = ::send(fd, send_data + sent, total - sent, MSG_NOSIGNAL);
      if (n > 0) sent += static_cast<size_t>(n);
      else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        std::perror("cross send"); std::exit(1);
      }
    }
    if ((event.revents & POLLIN) && received < total) {
      ssize_t n = ::recv(fd, recv_data + received, total - received, 0);
      if (n > 0) received += static_cast<size_t>(n);
      else if (n == 0) { std::fprintf(stderr, "cross peer closed\\n"); std::exit(1); }
      else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        std::perror("cross recv"); std::exit(1);
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(stderr, "usage: %s rank world gpu bytes iters local_ip remote_ip base_port\n", argv[0]);
    return 1;
  }
  int rank = std::atoi(argv[1]);
  int gpu = std::atoi(argv[3]);
  size_t bytes = std::strtoull(argv[4], nullptr, 10);
  int iters = std::atoi(argv[5]);
  const char* local_ip = argv[6];
  const char* remote_ip = argv[7];
  int base = std::atoi(argv[8]);
  size_t count = bytes / sizeof(float);

  CHECK(cudaSetDevice(gpu));
  float* device = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  std::vector<float> input(count), received(count), sum(count);
  std::vector<std::vector<float>> peer_data;
  for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(rank + 1);

  std::vector<int> workers;
  std::vector<int> crosses;
  if (rank == 0) {
    int local_server = listen_on(base);
    for (int i = 0; i < 3; ++i) {
      int fd = ::accept(local_server, nullptr, nullptr);
      int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      workers.push_back(fd);
    }
    ::close(local_server);
    int cross_server = listen_on(base + 8);
    for (int i = 0; i < 4; ++i) {
      int fd = ::accept(cross_server, nullptr, nullptr);
      int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      crosses.push_back(fd);
    }
    ::close(cross_server);
  } else if (rank == 4) {
    int remote_server = listen_on(base + 4);
    for (int i = 0; i < 7; ++i) {
      int fd = ::accept(remote_server, nullptr, nullptr);
      int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      workers.push_back(fd);
    }
    ::close(remote_server);
    for (int i = 0; i < 4; ++i) crosses.push_back(connect_to(local_ip, base + 8));
  } else if (rank < 4) {
    workers.push_back(connect_to(local_ip, base));
  } else {
    workers.push_back(connect_to(remote_ip, base + 4));
  }

  // Do not count connection/setup skew in the collective time.
  int ready = 1;
  if (rank == 0 || rank == 4) {
    for (int fd : workers) send_all(fd, &ready, sizeof(ready));
  } else {
    recv_all(workers[0], &ready, sizeof(ready));
  }

  if (rank == 0 || rank == 4) {
    peer_data.assign(workers.size(), std::vector<float>(count));
  }

  auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iters; ++iteration) {
    CHECK(cudaMemcpy(received.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (rank == 0 || rank == 4) {
      std::fill(sum.begin(), sum.end(), 0.0f);
      for (size_t i = 0; i < count; ++i) sum[i] += input[i];
      std::vector<std::thread> receivers;
      for (size_t p = 0; p < workers.size(); ++p) {
        receivers.emplace_back([&, p] {
          recv_all(workers[p], peer_data[p].data(), bytes);
        });
      }
      for (auto& thread : receivers) thread.join();
      for (size_t p = 0; p < workers.size(); ++p) {
        for (size_t i = 0; i < count; ++i) sum[i] += peer_data[p][i];
      }
      size_t part = count / 4;
      std::vector<std::thread> exchanges;
      for (int i = 0; i < 4; ++i) {
        size_t begin = i * part;
        size_t length = (i == 3) ? count - begin : part;
        exchanges.emplace_back([&, i, begin, length] {
          exchange_bidirectional(crosses[i], sum.data() + begin,
                                 received.data() + begin, length);
        });
      }
      for (auto& thread : exchanges) thread.join();
      for (size_t i = 0; i < count; ++i) sum[i] += received[i];
      std::vector<std::thread> senders;
      for (size_t p = 0; p < workers.size(); ++p) {
        senders.emplace_back([&, p] { send_all(workers[p], sum.data(), bytes); });
      }
      for (auto& thread : senders) thread.join();
      std::copy(sum.begin(), sum.end(), received.begin());
    } else {
      send_all(workers[0], input.data(), bytes);
      recv_all(workers[0], received.data(), bytes);
    }
    CHECK(cudaMemcpy(device, received.data(), bytes, cudaMemcpyHostToDevice));
  }
  auto end = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(end - start).count() / iters;
  CHECK(cudaMemcpy(received.data(), device, bytes, cudaMemcpyDeviceToHost));
  double expected = 78.0;
  int errors = 0;
  for (size_t i = 0; i < count; ++i) if (std::abs(received[i] - expected) > 1e-2) ++errors;
  double algbw = static_cast<double>(bytes) / seconds / 1e9;
  std::printf("rank=%d two-level bytes=%zu ms=%.3f algbw=%.3f errors=%d\n",
              rank, bytes, seconds * 1e3, algbw, errors);
  for (int fd : workers) ::close(fd);
  for (int fd : crosses) ::close(fd);
  CHECK(cudaFree(device));
  return errors == 0 ? 0 : 2;
}
