#include "plait/core/tensor.hpp"

namespace plait::core {

std::optional<std::string> validate_tensor_spec(const TensorSpec& tensor) {
  if (tensor.id.empty()) {
    return "tensor identity must not be zero";
  }
  if (tensor.dtype == DType::unknown) {
    return "tensor dtype must be known";
  }
  if (tensor.layout.kind == LayoutKind::strided &&
      tensor.layout.strides.size() != tensor.shape.size()) {
    return "strided layout must provide one stride per tensor dimension";
  }
  if (tensor.layout.kind != LayoutKind::strided &&
      !tensor.layout.strides.empty()) {
    return "only a strided layout may carry explicit strides";
  }
  return validate_placement(tensor.placement, tensor.shape.size());
}

}  // namespace plait::core
