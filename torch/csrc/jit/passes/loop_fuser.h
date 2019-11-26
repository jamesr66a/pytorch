#pragma once

#include <torch/csrc/jit/ir.h>

#include <memory>

namespace torch {
namespace jit {
namespace passes {

void TORCH_API LoopFuser(const std::shared_ptr<Graph>& graph);

}
} // namespace jit
} // namespace torch
