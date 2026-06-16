#ifndef MASSIM_GRADIENT_RESPONSE_H
#define MASSIM_GRADIENT_RESPONSE_H
#include <iostream>

#include "common.h"
#include <memory>
#include "distribution.h"
#include "rng.h"
// #include <omp.h>

#include <map>

namespace massim {

  MatType broadcast_cols(const ArrayType& vec, size_t n_rows) {
    return (VecType(vec) * Eigen::RowVectorXd::Constant(n_rows, 1.0)).array();
  }

  MatType broadcast_rows(const ArrayType& vec, size_t n_cols) {
    return broadcast_cols(vec, n_cols).transpose();
  }

// A GradientResponse represents a parameterized function that operates over
// a single gradient, combined with a set of parameters (usually drawn from
// a distribution.
class GradientResponse {

public:
  virtual ~GradientResponse() {}

  virtual MatType apply(const ArrayType &grad_coord) const = 0;

};

typedef std::unique_ptr<GradientResponse> GradientResponseUPtr;

class GradientResponseConfig {
public:
private:
  std::map<std::string, std::unique_ptr<Distribution>> m_params;

public:
  GradientResponseConfig() = default;
  virtual ~GradientResponseConfig()
    {}
  
  virtual std::unique_ptr<GradientResponse>
  generate_response(size_t N, RNG *rng = nullptr) const = 0;

protected:
  const Distribution &add_param_dist(const std::string &param_name,
                                     const DistUPtr &param_dist,
                                     const Distribution &default_dist) {
    if (param_dist == nullptr) {
      m_params[param_name] = default_dist.clone();
    } else {
      m_params[param_name] = param_dist->clone();
    }
    return *m_params[param_name];
  }

  const Distribution &add_param_dist(const std::string &param_name,
                                     const DistUPtr &param_dist) {
    m_params[param_name] = param_dist->clone();
    return *m_params[param_name];
  }

  std::map<std::string, ArrayType> sample_params(size_t N, RNG *rng) const {
    std::map<std::string, ArrayType> result;
    for (const auto &it : m_params) {
      result[it.first] = (*it.second)(N, rng);
    }
    return result;
  }
};

class BetaResponse : public GradientResponse {
  ArrayType m_mode;
  ArrayType m_range;
  ArrayType m_alpha;
  ArrayType m_gamma;

public:
  BetaResponse(const ArrayType &mode, const ArrayType &range,
               const ArrayType &alpha, const ArrayType &gamma):
  m_mode(mode), m_range(range), m_alpha(alpha), m_gamma(gamma)
  {}

  MatType apply(const ArrayType &X) const
  {
    
    ArrayType b = m_alpha / (m_alpha + m_gamma);
    ArrayType d = beta(b);
    ArrayType range_lo = m_mode - m_range * b;
    ArrayType range_hi = m_mode + m_range * (1-b);
    MatType result(m_alpha.size(), X.size());
    #pragma omp parallel shared(b,d,range_lo, range_hi, result, X)

    #pragma omp  for 
    for (int coord=0; coord < X.size(); ++coord) {
      ArrayType xn = (X[coord] - m_mode)/m_range + b;
      xn = ((X[coord] < range_lo) || (X[coord] > range_hi)).select(0, xn);
      xn = Eigen::pow(xn, m_alpha) * Eigen::pow(1-xn, m_gamma);

      result.col(coord) = beta(xn)/d;
	
    }
    return result;
  }

protected:
  MatType beta(const ArrayType &X) const {
    return Eigen::pow(X, m_alpha) * Eigen::pow(1-X, m_gamma);
  }
};

class BetaResponseConfig : public GradientResponseConfig {
  bool m_noskew;
  const Distribution &m_mode;
  const Distribution &m_range;
  const Distribution &m_alpha;
  const Distribution &m_gamma;

public:
  BetaResponseConfig(const DistUPtr &mode = nullptr,
                     const DistUPtr &range = nullptr,
                     const DistUPtr &alpha = nullptr,
                     const DistUPtr &gamma = nullptr, bool noskew = false)
      : m_noskew(noskew),
        m_mode(add_param_dist("mode", mode, UniformDistribution(-95, 195))),
        m_range(add_param_dist("range", range, NormalDistribution(100, 30))),
        m_alpha(add_param_dist("alpha", alpha, UniformDistribution(2.5, 6.5))),
        m_gamma(add_param_dist("gamma", gamma, UniformDistribution(2.5, 6.5))) {
  }

  std::unique_ptr<GradientResponse> generate_response(size_t N,
                                                      RNG *rng) const {
    auto sampled = sample_params(N, rng);
    auto alpha = sampled["alpha"];
    auto gamma = sampled["gamma"];
    if (m_noskew) {
      gamma = alpha;
    }
    return std::make_unique<BetaResponse>(sampled["mode"], sampled["range"],
                                          alpha, gamma);
  }
};

 
 GradientResponseUPtr test_massim() {
   BetaResponseConfig cfg(LinspaceDistribution(0,100).clone(),
			  ConstantDistribution(50).clone(),
			  ConstantDistribution(1).clone(),
			  ConstantDistribution(1).clone()
			  );
   auto z= cfg.generate_response(100, nullptr);
   return z;
 }

} // namespace massim

#endif // MASSIM_GRADIENT_RESPONSE_H
