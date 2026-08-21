#ifndef PLAIT_CORE_TENSOR_HPP_
#define PLAIT_CORE_TENSOR_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "plait/core/placement.hpp"

namespace plait::core {

enum class DType : std::uint8_t {
  unknown,
  boolean,
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  uint64,
  float16,
  bfloat16,
  float32,
  float64,
  complex64,
  complex128,
};

[[nodiscard]] constexpr std::string_view to_string(DType dtype) noexcept {
  switch (dtype) {
    case DType::unknown:
      return "unknown";
    case DType::boolean:
      return "bool";
    case DType::int8:
      return "int8";
    case DType::int16:
      return "int16";
    case DType::int32:
      return "int32";
    case DType::int64:
      return "int64";
    case DType::uint8:
      return "uint8";
    case DType::uint16:
      return "uint16";
    case DType::uint32:
      return "uint32";
    case DType::uint64:
      return "uint64";
    case DType::float16:
      return "float16";
    case DType::bfloat16:
      return "bfloat16";
    case DType::float32:
      return "float32";
    case DType::float64:
      return "float64";
    case DType::complex64:
      return "complex64";
    case DType::complex128:
      return "complex128";
  }
  return "unknown";
}

struct TensorId {
  std::uint64_t high{};
  std::uint64_t low{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return high == 0 && low == 0;
  }

  friend bool operator==(const TensorId&, const TensorId&) = default;
};

struct TensorVersion {
  std::uint64_t value{};

  friend bool operator==(const TensorVersion&, const TensorVersion&) = default;
};

using Shape = std::vector<std::uint64_t>;

enum class LayoutKind : std::uint8_t {
  contiguous,
  strided,
  opaque,
};

struct TensorLayout {
  LayoutKind kind{LayoutKind::contiguous};
  std::vector<std::int64_t> strides;
};

struct TensorSpec {
  TensorId id;
  TensorVersion version;
  Shape shape;
  DType dtype{DType::unknown};
  TensorLayout layout;
  Placement placement;
};

[[nodiscard]] std::optional<std::string> validate_tensor_spec(
    const TensorSpec& tensor);

}  // namespace plait::core

#endif  // PLAIT_CORE_TENSOR_HPP_
