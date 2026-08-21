#ifndef PLAIT_CORE_DEVICE_HPP_
#define PLAIT_CORE_DEVICE_HPP_

#include <cstdint>
#include <string>
#include <string_view>

namespace plait::core {

enum class BackendKind : std::uint8_t {
  cpu,
  cuda,
  hip,
  metal,
  opencl,
  pjrt,
  custom,
};

[[nodiscard]] constexpr std::string_view to_string(
    BackendKind backend) noexcept {
  switch (backend) {
    case BackendKind::cpu:
      return "cpu";
    case BackendKind::cuda:
      return "cuda";
    case BackendKind::hip:
      return "hip";
    case BackendKind::metal:
      return "metal";
    case BackendKind::opencl:
      return "opencl";
    case BackendKind::pjrt:
      return "pjrt";
    case BackendKind::custom:
      return "custom";
  }
  return "unknown";
}

struct NodeId {
  std::string value;

  friend bool operator==(const NodeId&, const NodeId&) = default;
};

struct DeviceId {
  NodeId node;
  BackendKind backend{BackendKind::cpu};
  std::uint32_t ordinal{};
  std::string stable_id;

  friend bool operator==(const DeviceId&, const DeviceId&) = default;
};

}  // namespace plait::core

#endif  // PLAIT_CORE_DEVICE_HPP_
