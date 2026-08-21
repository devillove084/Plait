#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "plait/core/placement.hpp"
#include "plait/core/tensor.hpp"

namespace {

using plait::core::BackendKind;
using plait::core::DeviceId;
using plait::core::NodeId;

[[nodiscard]] DeviceId device(std::string_view stable_id,
                              std::uint32_t ordinal) {
  return DeviceId{
      .node = NodeId{.value = "node-a"},
      .backend = BackendKind::cuda,
      .ordinal = ordinal,
      .stable_id = std::string{stable_id},
  };
}

TEST(Placement, ValidatesSharding) {
  using namespace plait::core;

  const auto gpu0 = device("gpu-0", 0);
  const auto gpu1 = device("gpu-1", 1);
  const Placement sharded = ShardedPlacement{
      .axis = 0,
      .devices = {gpu0, gpu1},
  };

  EXPECT_EQ(placement_kind(sharded), "sharded");
  EXPECT_FALSE(validate_placement(sharded, 2).has_value());
  EXPECT_TRUE(validate_placement(sharded, 0).has_value());
}

TEST(Placement, RejectsDuplicateDevices) {
  using namespace plait::core;

  const auto gpu0 = device("gpu-0", 0);
  const Placement duplicated = ReplicatedPlacement{.devices = {gpu0, gpu0}};

  EXPECT_TRUE(validate_placement(duplicated, 2).has_value());
}

TEST(TensorSpec, ValidatesLayout) {
  using namespace plait::core;

  Placement sharded = ShardedPlacement{
      .axis = 0,
      .devices = {device("gpu-0", 0), device("gpu-1", 1)},
  };
  TensorSpec tensor{
      .id = TensorId{.high = 0x1, .low = 0x2},
      .version = TensorVersion{.value = 0},
      .shape = {128, 256},
      .dtype = DType::float32,
      .layout = TensorLayout{.kind = LayoutKind::contiguous, .strides = {}},
      .placement = std::move(sharded),
  };

  EXPECT_FALSE(validate_tensor_spec(tensor).has_value());

  tensor.layout = TensorLayout{.kind = LayoutKind::strided, .strides = {1}};
  EXPECT_TRUE(validate_tensor_spec(tensor).has_value());
}

}  // namespace
