#ifndef MASSIM_SPECTRA_H
#define MASSIM_SPECTRA_H

#include <set>
#include "common.h"

namespace massim
{
  
typedef std::pair<MatType, ArrayType> SpecMatrix;
typedef std::vector<std::pair<int, double>> IntFloatVec;

class Spectra {
private:
  std::vector<IntFloatVec> m_samples;
  double m_tolerance;
public:
  Spectra() {}

  Spectra(const std::vector<IntFloatVec>& masslists,
	  double tolerance) :
    m_samples(masslists),
    m_tolerance(tolerance)
  {
    check_sorted();
  }

  Spectra(const ArrayType& masses,
	  const MatType& intensities,
	  double tolerance):
    m_tolerance(tolerance)
  {
    int nsamples = intensities.rows();
    int nmasses = masses.size();
    
    if (intensities.cols() != nmasses) {
      throw std::domain_error("Length of masses does not equal rows of intensity matrix.");
    }

    for (int samp_id=0; samp_id < nsamples; ++samp_id) {
      IntFloatVec sample;
      for (int mass_idx=0; mass_idx < nmasses; ++ mass_idx) {
	double intensity = intensities(samp_id, mass_idx);
	if (intensity > 0) {
	  int int_mass = int(masses(mass_idx) / tolerance);
	  sample.push_back(std::make_pair(int_mass, intensity));
	}
      }
      m_samples.emplace_back(sample);
    }
    check_sorted();
  }

  double tolerance() const {
    return m_tolerance;
  }

  void check_sorted() {
    for (auto &masslist : m_samples) {
      if (!std::is_sorted(masslist.begin(), masslist.end())) {
	std::sort(masslist.begin(), masslist.end());
      }
    }
  }
  
  const IntFloatVec& operator[](int idx) {
    return m_samples[idx];
  }

  int num_samples() const {
    return m_samples.size();
  }

  const std::vector<IntFloatVec>& data() const {
    return m_samples;
  }

  std::vector<int> masses() const {
    // Find all the masses in all the samples, and return as a sorted
    // vector.
    std::set<int> all_masses;
    for (auto sample : m_samples) {
      for (auto [mass, _] : sample) {
	all_masses.insert(mass);
      }
    }
    return std::vector<int>(all_masses.begin(), all_masses.end());
      
  }

  int mass_to_int(double mass) const {
    return int(mass / m_tolerance);
  }
  
  SpecMatrix to_matrix() const {
    auto int_masses = masses();

    
    MatType intensity(m_samples.size(), int_masses.size());
    for (size_t samp_id = 0; samp_id < m_samples.size(); ++samp_id) {
      const IntFloatVec& spec(m_samples[samp_id]);
      
      auto mass_idx = 0; // Index of mass in all masses

      for (auto [samp_mass, samp_val] :  spec) {
	while (samp_mass != int_masses[mass_idx]) {
	  // We know all spectal masses are in int_masses.
	  // As long as the spectra are sorted, this will work.
          mass_idx ++;
	}
	intensity(samp_id, mass_idx) = samp_val;
      }
    }

    ArrayType float_masses(int_masses.size());
    for (size_t mass_id = 0; mass_id < float_masses.size(); ++mass_id) {
      float_masses(mass_id) = int_masses[mass_id] * m_tolerance;
    }

    return SpecMatrix{intensity, float_masses};
  }
};



}  // namespace massim
#endif  // MASSIM_SPECTRA_H

