#include "metrics.h"
#include <stdexcept>
#include <algorithm>


namespace massim {

MatType compute_aff(const Eigen::Ref<const MatType>& sim) {
  int N = sim.rows();
  if (N != sim.cols()) {
    throw std::invalid_argument("Input matrix is not square");
  }
  const int max_rank_sum = (N-2) * (N-1) / 2;
  MatType result(N,N);
  
  for (int ii=0; ii<N; ++ii) {
    result(ii,ii) = 0.5;
    for (int jj=ii+1; jj<N; ++jj) {
      ArrayType diff = sim.col(ii) - sim.col(jj);
      ArrayType dij = diff.abs();
      // Flag ii and jj to ignore
      dij.coeffRef(ii) = std::numeric_limits<double>::max();
      dij.coeffRef(jj) = std::numeric_limits<double>::max();
      auto args = argsort(dij);
      double aff = 0;
      for (int kk=0; kk<N-2; kk++) {
	// Argsort gives the locations of the 1st, 2nd, 3rd rank items:
	if (diff.coeff(args[kk]) > 0)
	  aff += kk + 1;
      }
      aff /= max_rank_sum;
      result.coeffRef(ii, jj) = aff;
      result.coeffRef(jj, ii) = 1-aff;      
    }
  }
  return result;
}

MatType jaccard(const Eigen::Ref<const BoolMatType>& presence) {
  int N = presence.rows();
  int M = presence.cols();
  MatType result(M, M);
  IntArrayType total = presence.cast<int>().colwise().sum();
  for (int ii=0; ii < M; ++ii) {
    result(ii,ii) = 1.0;
    for (int jj=ii+1; jj< M; ++jj) {
      int zz = (presence.col(ii) && presence.col(jj)).cast<int>().sum();
      int denom = (total.coeff(ii) + total.coeff(jj) - zz);
      double dist = 1.0;
      if (denom > 0)
	dist = zz / (1.0 * denom);
      result.coeffRef(ii,jj) = dist;
      result.coeffRef(jj,ii) = dist;
    }
  }
  return result;
}

MatType jaccard(const Eigen::Ref<const MatType>& mat) {
  BoolMatType presence = mat > 0;
  return jaccard(presence);
}

MatType braycurtis(const Eigen::Ref<const MatType>& mat) {
  int N = mat.rows();
  int M = mat.cols();
  MatType result(M, M);
  ArrayType total = mat.colwise().sum();
  for (int ii=0; ii < M; ++ii) {
    result.coeffRef(ii,ii) = 1.0;
    for (int jj=ii+1; jj< M; ++jj) {
      double min_sum = 0;
      for (int kk=0; kk < N; ++kk)
	min_sum += std::min(mat.coeff(kk, ii), mat.coeff(kk, jj));
      double denom = total.coeff(ii) + total.coeff(jj);
      double dist = 1.0;
      if (denom > 0)
	dist = 2 * min_sum / denom;
      result.coeffRef(ii,jj) = dist;
      result.coeffRef(jj,ii) = dist;
    }
  }
  return result;
}



}  // namespace massim
