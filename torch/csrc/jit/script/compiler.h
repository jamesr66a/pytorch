#pragma once
#include <functional>
#include <memory>
#include <string>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/pybind.h"

namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(Graph*, std::string)>;

struct CompilationUnitImpl;
struct CompilationUnit {
  CompilationUnit(ResolutionCallback rcb);
  void define(const std::string& str);
  std::shared_ptr<Graph> getGraph(const std::string& func_name);
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

std::unique_ptr<CompilationUnit> jitScriptCompile(
    const std::string& script,
    ResolutionCallback rcb);

} // namespace script
} // namespace jit
} // namespace torch
