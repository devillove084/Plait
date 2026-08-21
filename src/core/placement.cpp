#include "plait/core/placement.hpp"

#include <cstddef>
#include <type_traits>
#include <unordered_set>

#include "absl/hash/hash.h"

namespace plait::core {
namespace {

template <class... Callables>
struct Overloaded : Callables... {
  using Callables::operator()...;
};

template <class... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

struct DeviceHash {
  [[nodiscard]] std::size_t operator()(const DeviceId& device) const {
    return absl::HashOf(device.node.value, device.backend, device.ordinal,
                        device.stable_id);
  }
};

[[nodiscard]] std::optional<std::string> validate_device(
    const DeviceId& device) {
  if (device.node.value.empty()) {
    return "device node identity must not be empty";
  }
  if (device.stable_id.empty()) {
    return "device stable identity must not be empty";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string> validate_device_set(
    const DeviceSet& devices) {
  if (devices.empty()) {
    return "device set must not be empty";
  }

  std::unordered_set<DeviceId, DeviceHash> seen_devices;
  seen_devices.reserve(devices.size());

  for (const auto& device : devices) {
    if (auto error = validate_device(device)) {
      return error;
    }
    if (!seen_devices.insert(device).second) {
      return "device set contains a duplicate stable identity";
    }
  }
  return std::nullopt;
}

}  // namespace

std::string_view placement_kind(const Placement& placement) {
  return std::visit(
      Overloaded{
          [](const LocalPlacement&) -> std::string_view { return "local"; },
          [](const ReplicatedPlacement&) -> std::string_view {
            return "replicated";
          },
          [](const ShardedPlacement&) -> std::string_view { return "sharded"; },
          [](const PartialPlacement&) -> std::string_view { return "partial"; },
      },
      placement);
}

std::optional<std::string> validate_placement(const Placement& placement,
                                              std::size_t tensor_rank) {
  return std::visit(Overloaded{
                        [](const LocalPlacement& local) {
                          return validate_device(local.device);
                        },
                        [](const ReplicatedPlacement& replicated) {
                          return validate_device_set(replicated.devices);
                        },
                        [tensor_rank](const ShardedPlacement& sharded) {
                          if (sharded.axis >= tensor_rank) {
                            return std::optional<std::string>{
                                "sharding axis is outside the tensor rank"};
                          }
                          return validate_device_set(sharded.devices);
                        },
                        [](const PartialPlacement& partial) {
                          return validate_device_set(partial.devices);
                        },
                    },
                    placement);
}

}  // namespace plait::core
