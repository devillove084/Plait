#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
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

class Barrier {
 public:
  explicit Barrier(int total) : total_(total) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int generation = generation_;
    if (++count_ == total_) {
      count_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != generation; });
    }
  }
 private:
  int total_;
  int count_ = 0;
  int generation_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
};

class Pipeline {
 public:
  Pipeline(int next_fd, int prev_fd)
      : next_fd_(next_fd), prev_fd_(prev_fd), start_(3), done_(3) {
    sender_ = std::thread([this] { sender_loop(); });
    receiver_ = std::thread([this] { receiver_loop(); });
  }
  ~Pipeline() {
    stop_.store(true);
    start_.arrive_and_wait();
    sender_.join();
    receiver_.join();
  }

  // Both send and receive run concurrently; the caller updates data after
  // both sides rendezvous. This removes the send-then-receive serialization.
  void step(const float* send_ptr, float* recv_ptr, size_t count,
            bool add_to_existing) {
    send_ptr_ = send_ptr;
    recv_ptr_ = recv_ptr;
    count_ = count;
    add_ = add_to_existing;
    start_.arrive_and_wait();
    done_.arrive_and_wait();
  }

  void copy_to(float* dst, const float* src, size_t count) const {
    std::memcpy(dst, src, count * sizeof(float));
  }

 private:
  void sender_loop() {
    while (true) {
      start_.arrive_and_wait();
      if (stop_.load()) return;
      send_all(next_fd_, send_ptr_, count_ * sizeof(float));
      done_.arrive_and_wait();
    }
  }
  void receiver_loop() {
    while (true) {
      start_.arrive_and_wait();
      if (stop_.load()) return;
      recv_all(prev_fd_, recv_ptr_, count_ * sizeof(float));
      done_.arrive_and_wait();
    }
  }

  int next_fd_;
  int prev_fd_;
  Barrier start_;
  Barrier done_;
  std::thread sender_;
  std::thread receiver_;
  const float* send_ptr_ = nullptr;
  float* recv_ptr_ = nullptr;
  size_t count_ = 0;
  bool add_ = false;
  std::atomic<bool> stop_{false};
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
  size_t width = (count + world - 1) / world;
  size_t padded = width * world;

  CHECK(cudaSetDevice(gpu));
  float* device = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  std::vector<float> data(padded, 0.0f), initial(padded, 0.0f), tmp(width), result(count);
  for (size_t i = 0; i < count; ++i) data[i] = static_cast<float>(rank + 1);
  initial = data;
  CHECK(cudaMemcpy(device, data.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(data.data(), device, bytes, cudaMemcpyDeviceToHost));

  auto ip_for = [&](int r) -> const char* { return r < 4 ? local_ip : remote_ip; };
  int listen_port = base_port + rank;
  int next_port = base_port + ((rank + 1) % world);
  const char* next_ip = ip_for((rank + 1) % world);

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
    data = initial;
    for (int step = 0; step < world - 1; ++step) {
      int send_chunk = (rank + world - step) % world;
      int recv_chunk = (rank + world - step - 1) % world;
      pipeline.step(data.data() + send_chunk * width, tmp.data(), width, true);
      for (size_t i = 0; i < width; ++i) data[recv_chunk * width + i] += tmp[i];
    }
    for (int step = 0; step < world - 1; ++step) {
      int send_chunk = (rank + world - step + 1) % world;
      int recv_chunk = (rank + world - step) % world;
      pipeline.step(data.data() + send_chunk * width, data.data() + recv_chunk * width, width, false);
    }
  }
  auto end = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(end - start).count() / iters;

  std::copy(data.begin(), data.begin() + count, result.begin());
  CHECK(cudaMemcpy(device, result.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost));
  double expected = static_cast<double>(world * (world + 1)) / 2.0;
  int errors = 0;
  for (size_t i = 0; i < count; ++i) if (std::abs(result[i] - expected) > 1e-2) ++errors;
  double algbw = static_cast<double>(bytes) / seconds / 1e9;
  double busbw = algbw * 2.0 * (world - 1) / world;
  std::printf("rank=%d world=%d gpu=%d bytes=%zu ms=%.3f algbw=%.3f busbw=%.3f errors=%d\n",
              rank, world, gpu, bytes, seconds * 1e3, algbw, busbw, errors);
  ::close(next_fd); ::close(prev_fd);
  CHECK(cudaFree(device));
  return errors == 0 ? 0 : 2;
}
