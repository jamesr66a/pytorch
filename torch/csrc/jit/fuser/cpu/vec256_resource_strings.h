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
  static constexpr int size() {
    return 32 / sizeof(T);
  }
  Vec256() {}
  Vec256(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  static Vec256<T> loadu(const void* ptr) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, 32);
    return vec;
  }
  static Vec256<T> loadu(const void* ptr, int64_t count) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(T));
    return vec;
  }
  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
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
  static Vec256<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align32__ float tmp_values[size()];
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm256_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_add_ps(a, b);
}

// ********** Vec256<int32_t> ***********

struct Vec256i {
protected:
  __m256i values;

  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1);
    return _mm256_xor_si256(ones, v);
  }
public:
  Vec256i() {}
  Vec256i(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
};

template <>
struct Vec256<int32_t> : public Vec256i {
  static constexpr int size() {
    return 8;
  }
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int32_t v) { values = _mm256_set1_epi32(v); }
  Vec256(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
         int32_t val5, int32_t val6, int32_t val7, int32_t val8) {
    values = _mm256_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  static Vec256<int32_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<int32_t> loadu(const void* ptr, int32_t count) {
    __at_align32__ int32_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(int32_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ int32_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
    }
  }
};

#endif // #if defined(__AVX__) && !defined(_MSC_VER)


}  // namespace

)");

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
