#ifndef PFFT_COMMON_H
#define PFFT_COMMON_H

#include <limits>
#include <exception>
#include <algorithm>
#include <numeric>

#include "Eigen/Dense"
#include "Eigen/Sparse"


namespace massim
{
  
const double NaN = std::numeric_limits<double>::quiet_NaN();
typedef Eigen::ArrayXd ArrayType;
typedef Eigen::VectorXd VecType;
// Use row major storage to align with numpy
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatType;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> BoolMatType;
typedef Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IntMatType;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolArrayType;
typedef Eigen::ArrayXi IntArrayType;
typedef Eigen::Ref<ArrayType> ArrayRef;
typedef Eigen::Ref<MatType> MatRef;
typedef Eigen::Ref<const ArrayType> ConstArrayRef;
typedef Eigen::Ref<const MatType> ConstMatRef;
typedef Eigen::Ref<const IntArrayType> ConstIntArrayRef;




class MassimException : public std::exception {
  std::string m_message;
public:
  MassimException(const std::string& msg) : m_message("Massim: " + msg)
  {}
  
  const char* what()  const noexcept override {
    return m_message.c_str();
  }
};


template <typename Type, class Compare = std::less<Type>>
inline std::vector<int> argsort(const Type* begin, int size, Compare comp = Compare())
{
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
        [&begin, &comp](int i1, int i2) { return comp(*(begin + i1), *(begin + i2)); });
    return idx;
}

// argsort for C++ std::vector
template <typename Type, class Compare = std::less<Type>>
inline std::vector<int> argsort(const std::vector<Type>& v, Compare comp = Compare())
{
    return argsort(&v[0], v.size(), comp);
}

// argsort for Eigen array
template <typename Type, class Compare = std::less<Type>>
inline std::vector<int> argsort(const Eigen::ArrayX<Type>& v, Compare comp = Compare())
{
    return argsort(&v[0], v.size(), comp);
}

  
}  // namespace massim
 
#endif  // MASSIM_COMMON_H
