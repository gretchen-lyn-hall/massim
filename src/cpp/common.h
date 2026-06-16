#ifndef PFFT_COMMON_H
#define PFFT_COMMON_H

#include <algorithm>
#include <exception>
#include <memory>
#include <limits>
#include <numeric>

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace massim {

const double NaN = std::numeric_limits<double>::quiet_NaN();
using ArrayType = Eigen::ArrayXd;
using VecType = Eigen::VectorXd;
// Use row major storage to align with numpy
using MatType =
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using BoolMatType =
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntMatType =
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using BoolArrayType = Eigen::Array<bool, Eigen::Dynamic, 1>;
using IntArrayType = Eigen::ArrayXi;
using ArrayRef = Eigen::Ref<ArrayType>;
using MatRef = Eigen::Ref<MatType>;
using ConstArrayRef = Eigen::Ref<const ArrayType>;
using ConstMatRef = Eigen::Ref<const MatType>;
using ConstIntArrayRef = Eigen::Ref<const IntArrayType>;

class MassimException : public std::exception {
  std::string m_message;

 public:
  MassimException(const std::string& msg) : m_message("Massim: " + msg) {}

  const char* what() const noexcept override { return m_message.c_str(); }
};

template <typename Type, class Compare = std::less<Type>>
inline std::vector<int> argsort(const Type* begin, int size,
                                Compare comp = Compare()) {
  std::vector<int> idx(size);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&begin, &comp](int i1, int i2) {
    return comp(*(begin + i1), *(begin + i2));
  });
  return idx;
}

// argsort for C++ std::vector
template <typename Type, class Compare = std::less<Type>>
inline std::vector<int> argsort(const std::vector<Type>& v,
                                Compare comp = Compare()) {
  return argsort(&v[0], v.size(), comp);
}

// argsort for Eigen array
template <typename Type, class Compare = std::less<Type>>
inline std::vector<int> argsort(const typename Eigen::ArrayX<Type>& v,
                                Compare comp = Compare()) {
  return argsort(&v[0], v.size(), comp);
}

}  // namespace massim

#endif  // MASSIM_COMMON_H
