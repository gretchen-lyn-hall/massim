#ifndef MASSIM_PROFILE_GENERATOR_H
#define MASSIM_PROFILE_GENERATOR_H

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "mass_transformer.h"
#include "mass_tracker.h"

namespace massim {

// Struct for keeping track of profiles whilst they are being built.
// Here, a "component" is the contribution that this profile will make
// to a particular peak; that is, a mass and abundance (called 'intensity'
// of a metabolite produced by this profile

struct ProfileComponents {
  MassTracker component_tracker;
  std::unordered_map<size_t, double> intensities;
  std::vector<size_t> mass_ids;

  bool update(double mass, size_t mass_id, double intensity) {
    if (intensities.contains(mass_id)) {
      intensities[mass_id] += intensity;
      return true;
    }
    intensities[mass_id] = intensity;
    mass_ids.push_back(mass_id);
    component_tracker.insert_mass(mass, /*force_insert=*/false,
                                  /*mass_id=*/mass_id);
    return true;
  }
};

struct ProfileResult {
  IntArrayType indices;
  ArrayType intensities;
};

class ProfileGenerator {
  MassTracker m_tracker;
  std::vector<ProfileComponents> m_profiles;
  double m_mass_min;
  double m_mass_max;
  std::unique_ptr<Distribution> m_iscale;
  double m_min_intensity;
  double m_preweight;
  double m_weight_exponent;
  ArrayType m_mass_deltas;
  ArrayType m_probs;
  ArrayType m_terminate_probs;
  std::unordered_set<size_t> m_terminated;
  bool m_enable_termination {false};
  ArrayType m_chain_lens;
  size_t m_stats_break_mass {};
  size_t m_stats_break_intens {};
  size_t m_stats_break_component {};
  std::vector<size_t> m_stats_applied;
  mutable ArrayType m_stats_weights;

 public:
  ProfileGenerator(size_t num_profiles, const TransformList &transforms,
                   const Distribution &intensity_scale, double mass_min = 150,
                   double mass_max = 1200, double min_intensity = 1e-6,
                   double thresh_ppm = 1.0, double preweight = 1.0,
                   double weight_exponent = 2.0);

  // Getter/setters for parameters
  const double preweight() const { return m_preweight; }
  void preweight(double new_val) {
    if (new_val <= 0) throw std::runtime_error("Preweight must be positive.");

    m_preweight = new_val;
  }

  double weight_exponent() const { return m_weight_exponent; }
  void weight_exponent(double new_val) {
    if (new_val < 0)
      throw std::runtime_error("Weight exponent must be nonnegative.");
    m_weight_exponent = new_val;
  }

  double mass_min() const { return m_mass_min; }
  void mass_min(double new_val) {
    if (new_val >= m_mass_max)
      throw std::runtime_error("Min mass must be less than max mass.");

    m_mass_min = new_val;
  }

  double mass_max() const { return m_mass_max; }
  void mass_max(double new_val) {
    if (new_val <= m_mass_min)
      throw std::runtime_error("Min mass must be less than max mass.");
    m_mass_max = new_val;
  }

  bool enable_termination() const { return m_enable_termination; }
  void enable_termination(bool new_val) { m_enable_termination = new_val; }

  // Getter/setter for term_probs
  const ArrayType &term_probs() const { return m_terminate_probs; }
  void term_probs(const ArrayType &new_val) {
    if (new_val.size() != m_probs.size()) {
      throw std::runtime_error(
          "Termination probabilities mush be same length "
          "as transform list.");
    }
    m_terminate_probs = new_val;
  }

  // Getters for computed results
  ArrayType masses() const { return m_tracker.masses(); }

  std::vector<ProfileResult> profiles() const;

  size_t num_masses() const { return m_tracker.size(); }

  size_t num_profiles() const { return m_profiles.size(); }

  // Weights used whilst selecting the last transformation
  ArrayType weights() const { return m_stats_weights; }

  // Counts of all transformations seen across all masses
  IntArrayType counts() const { return m_tracker.counts();}

  std::map<std::string, size_t> stats() const {
    
    std::map<std::string, size_t> result;
    result["break_mass"] = m_stats_break_mass;
    result["break_intensity"] = m_stats_break_intens;
    result["break_component"] = m_stats_break_component;
    for (int ii = 0; ii < m_stats_applied.size(); ++ii) {
      result[std::format("xfrm_{}", ii)] = m_stats_applied[ii];
    }
    return result;
  }

  // Add/update a mass for a given profile.
  // If mass already exists in profile, `intensity` is *added*.
  // Returns the actual mass inserted, and its ID.
  // The actual mass may differ if there is already an existing mass
  // (among all profiles) within tolerance.
  std::pair<double, size_t> update_component(size_t profile_id, double mass,
                                             double intensity);

  // The key function. Apply transformations randomly to the profiles
  // until the total number of metabolites exceeds `target_metabs`.

  void apply_transforms(size_t target_metabs, RNG &rng);

 private:
  // Return an xfrm, and number of times to apply forward/back, and whether
  // the new masses should be considered "dead ends"
  std::tuple<size_t, int, int, bool> choose_xfrm(RNG &rng) const;

  void extend_chain(size_t mass_id, size_t xfrm_id, size_t prof_id, int dircnt,
                    bool terminate, RNG &rng);
};

}  // namespace massim
#endif  // MASSIM_PROFILE_GENERATOR_H
