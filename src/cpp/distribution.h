#ifndef MASSIM_DISTRIBUTION_H
#define MASSIM_DISTRIBUTION_H

#include <random>
#include <stdexcept>
#include <vector>
#include <format>

#include "common.h"
#include "rng.h"

namespace massim
{    

class Distribution {
  std::vector<double> m_default_buf;
  
public:
  Distribution():
    m_default_buf(1)
  {}

  Distribution(const Distribution&) = default;


  virtual ~Distribution(){
  }

  virtual std::string name() const = 0;

  virtual std::vector<double> params() const = 0;

  virtual std::unique_ptr<Distribution> clone() const = 0;
  
  
  double operator()() const{
    return this->sample_one(RNG::default_rng());
  }

  double operator()(RNG& rng) const{
    return this->sample_one(rng);
  }

  template <class IterT>
  void operator()(const IterT& begin,
		  const IterT& end,
		  RNG* rng=nullptr) const{
    if (rng == nullptr) {
      rng = &RNG::default_rng();
    }
    size_t N = std::distance(begin, end);
    ArrayType result = sample_n(N, *rng);
      
    std::generate(result.begin(), result.end(), begin);
  }
  
  ArrayType operator()(size_t count, RNG* rng=nullptr) {
    if (rng == nullptr) {
      rng = &RNG::default_rng();
    }
    return sample_n(count, *rng);
      
  }

protected:
  virtual double sample_one(RNG& rng) const = 0;
  virtual ArrayType sample_n(size_t count, RNG& rng) const {
    ArrayType result {count};
    for (size_t ii=0; ii<count; ++ii) {
      result.coeffRef(ii) = sample_one(rng);
    }
    return result;
  }

};

typedef std::unique_ptr<Distribution> DistUPtr;


 
 class ConstantDistribution : public Distribution {
   double m_val;
 public:
 ConstantDistribution(double val):
   m_val(val)
  {}

   std::string name() const {
     return "Constant";
   }

   std::vector<double> params() const {
     return {m_val};
   }

   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<ConstantDistribution>(*this);
   }

protected:
  double sample_one(RNG& rng) const {
    return m_val;
  }
  
  ArrayType sample_n(size_t count, RNG& rng) const {
    return ArrayType::Constant(count, m_val);
  }
  
};


 class UniformDistribution : public Distribution {
   mutable std::uniform_real_distribution<double> m_dist;
public:
  UniformDistribution(double low, double high):
    m_dist(low, high)
  {}

    std::string name() const {
     return "UniformDistribution";
   }

   std::vector<double> params() const {
     return {m_dist.param().a(), m_dist.param().b()};
   }
   
   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<UniformDistribution>(*this);
   }


 protected:
  double sample_one(RNG& rng) const {
    return m_dist(rng.engine);
  }

   ArrayType sample_n(size_t count, RNG& rng) const {
     ArrayType result {count};
     for (size_t ii=0; ii<count; ++ii) {
       result.coeffRef(ii) = m_dist(rng.engine);
     }
     return result;
   }

};

class NormalDistribution : public Distribution {
  mutable std::normal_distribution<double> m_dist;
public:
  NormalDistribution(double mean, double std):
    m_dist(mean, std)
  {}


    std::string name() const {
     return "NormalDistribution";
   }

   std::vector<double> params() const {
     return {m_dist.param().mean(), m_dist.param().stddev()};
   }

   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<NormalDistribution>(*this);
   }

protected:
  double sample_one(RNG& rng) const {
    return m_dist(rng.engine);
  }
};

class BetaDistribution : public Distribution {
  mutable std::gamma_distribution<double> m_X;
  mutable std::gamma_distribution<double> m_Y;
public:
    BetaDistribution(double a, double b)
      : m_X(a),
	m_Y(b)
  {}


    std::string name() const {
     return "BetaDistribution";
   }

   std::vector<double> params() const {
     return {m_X.param().alpha(), m_Y.param().alpha()};
   }

   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<BetaDistribution>(*this);
   }

protected:
  double sample_one(RNG &rng) const {
    double X = m_X(rng.engine);
    return X / (X + m_Y(rng.engine));
  }
};



class LognormalDistribution : public Distribution {
  mutable std::lognormal_distribution<double> m_dist;
public:
  LognormalDistribution(double mean, double std):
    m_dist(mean, std)
  {}

    std::string name() const {
     return "LogNormalDistribution";
   }

   std::vector<double> params() const {
     return {m_dist.param().m(), m_dist.param().s()};
   }

   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<LognormalDistribution>(*this);
   }


protected:
  double sample_one(RNG& rng) const {
    return m_dist(rng.engine);
  }
};


class BinOpDistribution : public Distribution {
  std::unique_ptr<Distribution> m_lhs;
  std::unique_ptr<Distribution> m_rhs;
  char m_op_char;
  ArrayType(*m_op)(const ArrayType&, const ArrayType&);

  static ArrayType add_op(const ArrayType& lhs, const ArrayType& rhs) {
    return lhs+rhs;
  }

  static ArrayType sub_op(const ArrayType& lhs, const ArrayType& rhs) {
    return lhs-rhs;
  }

  static ArrayType mul_op(const ArrayType& lhs, const ArrayType& rhs) {
    return lhs*rhs;
  }

  static ArrayType div_op(const ArrayType& lhs, const ArrayType& rhs) {
    return lhs/rhs;
  }

public:
  BinOpDistribution(std::unique_ptr<Distribution> lhs,
		    std::unique_ptr<Distribution> rhs,
		    char op):
  m_lhs(std::move(lhs)), m_rhs(std::move(rhs)), m_op_char(op)
    {
      switch (op) {
      case '+':
	m_op = BinOpDistribution::add_op;
	break;
      case '-':
	m_op = BinOpDistribution::sub_op;
	break;
      case '*':
	m_op = BinOpDistribution::mul_op;
	break;
      case '/':
	m_op = BinOpDistribution::div_op;
	break;
      default:
	throw MassimException("Unknown operator");
      }
    }	

  std::string name() const {
    return std::format("BinaryOperator:{}", m_op_char);
  }

   std::vector<double> params() const {
     return {};
   }

   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<BinOpDistribution>(m_lhs->clone(),
						m_rhs->clone(),
						m_op_char);
   }


protected:
  double sample_one(RNG& rng) const {
    double lhs = (*m_lhs)(rng);
    double rhs = (*m_rhs)(rng);
    switch (m_op_char) {
    case '+':
      return lhs + rhs;
    case '-':
      return lhs - rhs;
    case '*':
      return lhs * rhs;
    case '/':
      return lhs / rhs;      
    }
    throw std::logic_error("Never should I be here");
  }
  
  ArrayType sample_n(size_t count, RNG& rng) const {
    ArrayType lhs = (*m_lhs)(count, &rng);
    ArrayType rhs = (*m_rhs)(count, &rng);
    return m_op(lhs, rhs);
  }

};

class UnOpDistribution : public Distribution {
  std::unique_ptr<Distribution> m_lhs;
  char m_op_char;
public:
  UnOpDistribution(std::unique_ptr<Distribution> lhs,
		   char op):
  m_lhs(std::move(lhs)), m_op_char(op)
    {
      if (op != '-') {
	throw MassimException(std::format("Unknown unary op '{}'", op));
      }
    }	
    
  std::string name() const {
    return std::format("UnaryOperator:{}", m_op_char);
  }

  std::vector<double> params() const {
    return {};
  }

  std::unique_ptr<Distribution> clone() const {
    return std::make_unique<UnOpDistribution>(m_lhs->clone(),
					      m_op_char);
  }

 protected:
  double sample_one(RNG& rng) const {
    double lhs = (*m_lhs)(rng);
    return -lhs;
  }
  
  ArrayType sample_n(size_t count, RNG& rng) const {
    ArrayType lhs = (*m_lhs)(count, &rng);
    return -lhs;
  }

};

 
// A pseudo-distribution. Returns evenly spaced values between min and max.
class LinspaceDistribution : public Distribution {
  double m_min;
  double m_max;
public:
 LinspaceDistribution(double min, double max):
  m_min(min), m_max(max)
  {}

  std::string name() const {
    return "LinspaceDistribution";
  }

   std::vector<double> params() const {
     return {m_min, m_max};
   }

   std::unique_ptr<Distribution> clone() const {
     return std::make_unique<LinspaceDistribution>(*this);
   }

 protected:
  double sample_one(RNG& rng) const {
    throw MassimException("Linspace distribition must be sampled as a vector");
  }
  
  ArrayType sample_n(size_t count, RNG& rng) const {
    return ArrayType::LinSpaced(count, m_min, m_max);
  }
};


std::unique_ptr<Distribution> gen_dist(
				       const std::string& dist_type,
					   const std::vector<double> &args);

std::unique_ptr<Distribution> parse_dist(const std::string& dist_type);

}  // namespace massim

#endif  // MASSIM_DISTRIBUTION_H
