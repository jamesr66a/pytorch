#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

// Returns true if the node is added to the fusion group, false o.w.
TORCH_API bool mergeNodeWithFusionGroup(const Node* const node, Node* fusion_group);

}}}} // namespace torch::jit::fuser::cpu
