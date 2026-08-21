#ifndef PLAIT_CORE_PLACEMENT_HPP_
#define PLAIT_CORE_PLACEMENT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "plait/core/device.hpp"

namespace plait::core {

using DeviceSet = std::vector<DeviceId>;

struct LocalPlacement {
  DeviceId device;
};

struct ReplicatedPlacement {
  DeviceSet devices;
};

struct ShardedPlacement {
  std::size_t axis{};
  DeviceSet devices;
};

enum class ReductionOp : std::uint8_t {
  sum,
  product,
  minimum,
  maximum,
};

struct PartialPlacement {
  ReductionOp operation{ReductionOp::sum};
  DeviceSet devices;
};

using Placement = std::variant<LocalPlacement, ReplicatedPlacement,
                               ShardedPlacement, PartialPlacement>;

[[nodiscard]] std::string_view placement_kind(const Placement& placement);

// Returns no value when the placement is valid. A tensor rank is supplied so
// that sharding axes can be checked without coupling placement to TensorSpec.
[[nodiscard]] std::optional<std::string> validate_placement(
    const Placement& placement, std::size_t tensor_rank);

}  // namespace plait::core

#endif  // PLAIT_CORE_PLACEMENT_HPP_
