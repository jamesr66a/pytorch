#pragma once

#include <torch/csrc/jit/code_template.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

static auto vec256_template = CodeTemplate(R"(
// ********** Vec256 **********
#if defined(__GNUC__)
#define __at_align32__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align32__ __declspec(align(32))
#else
#define __at_align32__
#endif

namespace {
template<size_t n> struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

template <class T>
struct Vec256 {
private:
  T values[32 / sizeof(T)] = {0};
public:
  constexpr int size() {
    return 32 / sizeof(T);
  }
  Vec256() {}
  Vec256(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
};

template <class T> Vec256<T> inline operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}


// ********** Vec256<float> **********
#if defined(__AVX__) && !defined(_MSC_VER)

#include <x86intrin.h>

template <> class Vec256<float> {
private:
  __m256 values;
public:
  static constexpr int size() {
    return 8;
  }
  Vec256() {}
  Vec256(__m256 v) : values(v) {}
  Vec256(float val) {
    values = _mm256_set1_ps(val);
  }
  Vec256(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8) {
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  operator __m256() const {
    return values;
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_add_ps(a, b);
}

#endif // #if defined(__AVX__) && !defined(_MSC_VER)


}  // namespace

)");

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
