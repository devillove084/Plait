#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <fcntl.h>
#include <sys/epoll.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
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

class Pipeline {
 public:
  Pipeline(int next_fd, int prev_fd) : next_fd_(next_fd), prev_fd_(prev_fd) {
    epoll_ = ::epoll_create1(0);
    if (epoll_ < 0) { std::perror("epoll_create1"); std::exit(1); }
    auto nonblock = [](int fd) {
      int flags = ::fcntl(fd, F_GETFL, 0);
      if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        std::perror("fcntl"); std::exit(1);
      }
    };
    nonblock(next_fd_);
    nonblock(prev_fd_);
    epoll_event event{};
    event.events = EPOLLOUT;
    event.data.fd = next_fd_;
    if (::epoll_ctl(epoll_, EPOLL_CTL_ADD, next_fd_, &event) < 0) {
      std::perror("epoll_ctl next"); std::exit(1);
    }
    event.events = EPOLLIN;
    event.data.fd = prev_fd_;
    if (::epoll_ctl(epoll_, EPOLL_CTL_ADD, prev_fd_, &event) < 0) {
      std::perror("epoll_ctl prev"); std::exit(1);
    }
  }

  ~Pipeline() { ::close(epoll_); }

  void step(float* send_ptr, size_t send_count, float* recv_ptr, size_t recv_count) {
    const char* send_data = reinterpret_cast<const char*>(send_ptr);
    char* recv_data = reinterpret_cast<char*>(recv_ptr);
    size_t send_size = send_count * sizeof(float);
    size_t recv_size = recv_count * sizeof(float);
    size_t sent = 0;
    size_t received = 0;
    while (sent < send_size || received < recv_size) {
      epoll_event events[2];
      int ready = ::epoll_wait(epoll_, events, 2, -1);
      if (ready < 0) {
        if (errno == EINTR) continue;
        std::perror("epoll_wait");
        std::exit(1);
      }
      for (int i = 0; i < ready; ++i) {
        if (events[i].data.fd == next_fd_ && sent < send_size) {
          ssize_t n = ::send(next_fd_, send_data + sent, send_size - sent, MSG_NOSIGNAL);
          if (n > 0) sent += static_cast<size_t>(n);
          else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            std::perror("send"); std::exit(1);
          }
        } else if (events[i].data.fd == prev_fd_ && received < recv_size) {
          ssize_t n = ::recv(prev_fd_, recv_data + received, recv_size - received, 0);
          if (n > 0) received += static_cast<size_t>(n);
          else if (n == 0) { std::fprintf(stderr, "recv: peer closed\n"); std::exit(1); }
          else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            std::perror("recv"); std::exit(1);
          }
        }
      }
    }
  }

 private:
  int next_fd_;
  int prev_fd_;
  int epoll_;
};

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(stderr,
                 "usage: %s rank world gpu bytes iters local_ip remote_ip base_port\n",
                 argv[0]);
    return 1;
  }
  int rank = std::atoi(argv[1]);
  int world = std::atoi(argv[2]);
  int gpu = std::atoi(argv[3]);
  size_t bytes = std::strtoull(argv[4], nullptr, 10);
  int iters = std::atoi(argv[5]);
  const char* local_ip = argv[6];
  const char* remote_ip = argv[7];
  int base_port = std::atoi(argv[8]);
  size_t count = bytes / sizeof(float);
  constexpr int kRowsTotal = 64;
  if (count % kRowsTotal != 0) {
    std::fprintf(stderr, "bytes must describe 64 rows\n");
    return 1;
  }
  size_t cols = count / kRowsTotal;
  const int rows[12] = {2, 2, 2, 2, 7, 7, 7, 7, 7, 7, 7, 7};
  size_t local_count = rows[rank] * cols;
  auto ip_for = [&](int r) -> const char* { return r < 4 ? local_ip : remote_ip; };
  const int order[12] = {0, 4, 1, 5, 2, 6, 3, 7, 8, 9, 10, 11};
  int position = 0;
  for (int i = 0; i < 12; ++i) if (order[i] == rank) position = i;
  int listen_port = base_port + rank;

  CHECK(cudaSetDevice(gpu));
  float* device = nullptr;
  CHECK(cudaMalloc(&device, local_count * sizeof(float)));
  std::vector<float> local(local_count, static_cast<float>(rank + 1));
  CHECK(cudaMemcpy(device, local.data(), local_count * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(local.data(), device, local_count * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> data(count, 0.0f), tmp;
  std::vector<int> peers;
  std::vector<size_t> chunk_size(12), chunk_offset(12);
  size_t assigned = 0;
  for (int i = 0; i < 12; ++i) {
    chunk_offset[i] = assigned;
    chunk_size[i] = rows[order[i]] * cols;
    assigned += chunk_size[i];
  }
  if (assigned != count) {
    std::fprintf(stderr, "chunk sizes do not cover count\n");
    return 1;
  }
  size_t max_chunk = 0;
  for (size_t size : chunk_size) max_chunk = std::max(max_chunk, size);
  tmp.assign(max_chunk, 0.0f);
  std::copy(local.begin(), local.end(), data.begin() + chunk_offset[position]);
  int next_rank = order[(position + 1) % 12];
  int next_port = base_port + next_rank;
  const char* next_ip = ip_for(next_rank);

  int server = ::socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in bind_address{};
  bind_address.sin_family = AF_INET;
  bind_address.sin_addr.s_addr = INADDR_ANY;
  bind_address.sin_port = htons(listen_port);
  if (::bind(server, reinterpret_cast<sockaddr*>(&bind_address), sizeof(bind_address)) < 0) {
    std::perror("bind"); return 1;
  }
  if (::listen(server, 1) < 0) { std::perror("listen"); return 1; }

  int next_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int nodelay = 1;
  ::setsockopt(next_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  sockaddr_in next_address{};
  next_address.sin_family = AF_INET;
  next_address.sin_port = htons(next_port);
  ::inet_pton(AF_INET, next_ip, &next_address.sin_addr);
  for (int attempt = 0; attempt < 240; ++attempt) {
    if (::connect(next_fd, reinterpret_cast<sockaddr*>(&next_address), sizeof(next_address)) == 0) break;
    ::close(next_fd); ::usleep(250000);
    next_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ::setsockopt(next_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    if (attempt == 239) { std::perror("connect"); return 1; }
  }
  int prev_fd = ::accept(server, nullptr, nullptr);
  if (prev_fd < 0) { std::perror("accept"); return 1; }
  ::setsockopt(prev_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  ::close(server);

  auto start = std::chrono::steady_clock::now();
  Pipeline pipeline(next_fd, prev_fd);
  for (int iteration = 0; iteration < iters; ++iteration) {
    // Ring all-gather of unequal local partials. Chunks are already sized by
    // the LP row assignment; no send length is inferred from world size.
    for (int step = 0; step < world - 1; ++step) {
      int send_chunk = (position + world - step) % world;
      int recv_chunk = (position + world - step - 1) % world;
      pipeline.step(data.data() + chunk_offset[send_chunk], chunk_size[send_chunk],
                    tmp.data(), chunk_size[recv_chunk]);
      std::copy(tmp.begin(), tmp.begin() + chunk_size[recv_chunk],
                data.begin() + chunk_offset[recv_chunk]);
    }
  }
  auto end = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(end - start).count() / iters;

  int errors = 0;
  for (int pos = 0; pos < 12; ++pos) {
    int owner = order[pos];
    float wanted = static_cast<float>(owner + 1);
    for (size_t i = 0; i < chunk_size[pos]; ++i) {
      if (std::abs(data[chunk_offset[pos] + i] - wanted) > 1e-2) ++errors;
    }
  }
  double algbw = static_cast<double>(bytes) / seconds / 1e9;
  double busbw = algbw * 2.0 * 11.0 / 12.0;
  std::printf("rank=%d world=12 gpu=%d bytes=%zu ms=%.3f algbw=%.3f busbw=%.3f errors=%d\\n",
              rank, gpu, bytes, seconds * 1e3, algbw, busbw, errors);
  ::close(next_fd); ::close(prev_fd);
  CHECK(cudaFree(device));
  return errors == 0 ? 0 : 2;

}
