#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <aten/src/ATen/core/jit_type.h>

namespace torch {
namespace jit {
namespace fuser {

TORCH_API size_t getRank(const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API size_t getNumel(const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API size_t getNumNonCollapsibleDims(const std::shared_ptr<c10::TensorType>& tensor);

}}} // namespace torch::jit::fuser
