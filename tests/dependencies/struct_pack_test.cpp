#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <ylt/struct_pack.hpp>

namespace {

struct WireDevice {
  std::uint64_t node{};
  std::string backend;
  std::uint32_t ordinal{};

  friend bool operator==(const WireDevice&, const WireDevice&) = default;
};

TEST(DependencyContract, StructPackRoundTripsWireMetadata) {
  const WireDevice original{
      .node = 0x1234,
      .backend = "opencl",
      .ordinal = 7,
  };

  const auto encoded = struct_pack::serialize(original);
  const auto decoded = struct_pack::deserialize<WireDevice>(encoded);

  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded.value(), original);
}

}  // namespace
