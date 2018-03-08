#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/ScalarType.h"

namespace at {
namespace native {

#define DEFINE_CAST_OP(_1, n, _2)                                            \
  Tensor cast_##_1(const Tensor& self, bool non_blocking) {                  \
    return self.type().toScalarType(ScalarType::n).copy(self, non_blocking); \
  }

AT_FORALL_SCALAR_TYPES(DEFINE_CAST_OP)

#undef DEFINE_CAST_OP

Tensor empty_like(const Tensor& self) {
  return self.type().tensor(self.sizes());
}

Tensor empty_like(const Tensor& self, const Type& dtype) {
  return dtype.tensor(self.sizes());
}

Tensor ones_like(const Tensor& self) {
  return self.type().ones(self.sizes());
}

Tensor ones_like(const Tensor& self, const Type& dtype) {
  return dtype.ones(self.sizes());
}

Tensor rand_like(const Tensor& self) {
  return self.type().rand(self.sizes());
}

Tensor rand_like(const Tensor& self, const Type& dtype) {
  return dtype.rand(self.sizes());
}

Tensor randn_like(const Tensor& self) {
  return self.type().randn(self.sizes());
}

Tensor randn_like(const Tensor& self, const Type& dtype) {
  return dtype.randn(self.sizes());
}

Tensor zeros_like(const Tensor& self) {
  return at::native::zeros_like(self, self.type());
}

Tensor zeros_like(const Tensor& self, const Type& dtype) {
  if (dtype.is_sparse() && self.type().is_sparse()) {
    auto res = dtype.tensor();
    res.resize_as_(self);
    res.zero_();
    return res;
  }
  return dtype.zeros(self.sizes());
}

}
}
