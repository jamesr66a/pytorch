#pragma once

#include <aten/src/ATen/core/ivalue.h>

namespace torch {
namespace jit {
namespace script {

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;

struct TORCH_API Object {
  Object() {}
  Object(ObjectPtr object_value) : object_value_(std::move(object_value)) {}
  explicit Object(c10::QualifiedName class_name);
  Object(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);
  Object(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);

  ObjectPtr object_value() const;

 private:
  // mutable be we lazily initialize in module_object.
  mutable ObjectPtr object_value_;
};

} // namespace script
} // namespace jit
} // namespace torch
