#include "profile_generator.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace massim {

ProfileGenerator::ProfileGenerator(size_t num_profiles,
                                   const TransformList &transforms,
                                   const Distribution &intensity_scale,
                                   double mass_min, double mass_max,
                                   double min_intensity, double thresh_ppm,
                                   double preweight, double weight_exponent)
    : m_tracker(transforms.mass_deltas(), thresh_ppm, MODE_STRICT, false,
                false),
      m_iscale(std::move(intensity_scale.clone())),
      m_mass_min(mass_min),
      m_mass_max(mass_max),
      m_min_intensity(min_intensity),
      m_preweight(preweight),
      m_weight_exponent(weight_exponent),
      m_mass_deltas(transforms.mass_deltas()),
      m_probs(transforms.raw_probabilities()),
      m_enable_termination(false),
      m_terminate_probs((m_probs.maxCoeff() - m_probs) / m_probs.maxCoeff()),
      m_chain_lens(transforms.chain_lengths()),
      m_stats_break_mass(0),
      m_stats_break_intens(0),
      m_stats_applied(transforms.num_xfrms(), 0) {
  // Initialize the tracker and intensity maps of all the profiles.
  for (int ii = 0; ii < num_profiles; ++ii) {
    m_profiles.push_back({
        MassTracker(transforms.mass_deltas(), thresh_ppm, MODE_STRICT, false,
                    false)
        // Default constructor for abundances
    });
  }
}

std::pair<double, size_t> ProfileGenerator::update_component(size_t profile_id,
                                                             double mass,
                                                             double intensity) {
  if (profile_id >= m_profiles.size()) {
    throw std::runtime_error("Profile id out of range");
  }
  size_t mass_id = m_tracker.insert_mass(mass);
  double actual_mass = m_tracker.find_mass_by_id(mass_id);
  m_profiles[profile_id].update(actual_mass, mass_id, intensity);
  return {actual_mass, mass_id};
}

void ProfileGenerator::apply_transforms(size_t target_metabs, RNG &rng) {
  std::uniform_int_distribution<size_t> choose_prof(0, m_profiles.size() - 1);

  // Choose the first transform
  auto [xfrm_id, fwd, back, terminate] = choose_xfrm(rng);

  while (m_tracker.size() < target_metabs) {
    // First, randomly pick which profile to modify. This is just uniformly
    // chosen.
    size_t prof_id = choose_prof(rng.engine);
    ProfileComponents &prof = m_profiles[prof_id];


    // This differs a bit from the python code; there we shuffle the component
    // ids, then iterate through them. However, that is *terribly* slow in c++
    // for whatever reason. Instead, we keep randomly picking components
    // instead until we find a workable one that
    // - Hasn't yet had the transform applied to both sides
    // - Hasn't been marked as a dead end ('terminated')
    // If we can't find one in a reasonable number of guesses, we'll abort
    // this loop.
    
    std::uniform_int_distribution<> choose_comp(0, prof.mass_ids.size()-1);
    int comp_id = -1;
    for (int idx=0; idx < prof.mass_ids.size(); ++idx) {
      size_t test_id = prof.mass_ids[choose_comp(rng.engine)];
      if (m_terminated.contains(test_id)) {
	continue;
      }
      auto [src, dst] =
          prof.component_tracker.contains_xfrm_by_id(test_id, xfrm_id);
      if (!(src && dst) ) {
        comp_id = test_id;
        break;
      }
    }

    if (comp_id >= 0 ) {
      // We should pretty much always be able to find a component above, in
      // which case we apply it!
      extend_chain(comp_id, xfrm_id, prof_id, fwd, terminate, rng);
      extend_chain(comp_id, xfrm_id, prof_id, -back, terminate, rng);

      // Transform successfully applied; choose the next
      std::tie(xfrm_id, fwd, back, terminate) = choose_xfrm(rng);

      
    } else {
      // Else: try again with a new profile/transformation
      m_stats_break_component += 1;
    }
  }
}

std::vector<ProfileResult> ProfileGenerator::profiles() const {
  // The mass_ids assigned by the tracker are in arbitrary order;
  // we want to return the ids as their position in the sorted mass list.

  std::vector<size_t> index_lookup(m_tracker.size());
  auto sorted_ids = m_tracker.mass_ids();
  // Here we rely on the fact that the id's assigned by the tracker
  // are all within range [0, N).
  for (int ii = 0; ii < m_tracker.size(); ++ii) {
    index_lookup[sorted_ids[ii]] = ii;
  }

  std::vector<ProfileResult> result;
  for (const auto &prof : m_profiles) {
    ProfileResult p_result{IntArrayType::Zero(prof.intensities.size()),
                           ArrayType::Zero(prof.intensities.size())};
    std::vector<std::pair<size_t, double>> indices;
    // First, remap the raw_id/intensity pairs to ordered mass ids
    std::transform(prof.intensities.begin(), prof.intensities.end(),
                   std::back_inserter(indices), [&](const auto &it) {
                     return std::make_pair(index_lookup[it.first], it.second);
                   });
    // Sort by actual mass order
    std::sort(indices.begin(), indices.end());

    // Fill in results
    for (size_t ii = 0; ii < indices.size(); ++ii) {
      p_result.indices(ii) = indices[ii].first;
      p_result.intensities(ii) = indices[ii].second;
    }
    result.emplace_back(p_result);
  }
  return result;
}

std::tuple<size_t, int, int, bool> ProfileGenerator::choose_xfrm(
    RNG &rng) const {
  static std::uniform_real_distribution<> choose_term(0, 1);
  const IntArrayType counts = m_tracker.counts();

  // Find ratio of how many of each xfrm we have seen to the raw prob
  // (with preweight to balance things out when total_count() is small)
  ArrayType cur_frac = (counts.cast<double>() + m_preweight * m_probs) /
                       (m_preweight + m_tracker.total_count());
  // Get rid of any zeros, which can only happen if the original probs
  // are zero (or preweight is zero)
  cur_frac = (cur_frac.array() == 0).select(1.0, cur_frac);
  ArrayType weights = (m_probs / cur_frac).pow(m_weight_exponent);
  m_stats_weights = weights;

  // Now choose an id based on `desired_prob * weight`
  ArrayType weighted_probs = weights * m_probs;
  std::discrete_distribution<> xfrm_choice(weighted_probs.begin(),
                                           weighted_probs.end());

  // Choose how many times to apply it. The total chain length is given by
  // a negative binomial, from which we uniformly randomly split into
  // forward and back.
  size_t xfrm_id = xfrm_choice(rng.engine);
  int total_len = 1 + negative_binomial(m_chain_lens(xfrm_id) - 1.999, rng);
  int fwd = std::uniform_int_distribution<>(0, total_len)(rng.engine);

  // For rare transforms, we have to be careful. If we add a rare transform
  // early on (say between A and B), then whenever we add a common transform
  // (e.g.+CH2) to both A and B, it will increase the rare transform count.
  bool terminate = m_terminate_probs(xfrm_id) > choose_term(rng.engine);
  return {xfrm_id, fwd, total_len - fwd, terminate};
}

void ProfileGenerator::extend_chain(size_t mass_id, size_t xfrm_id,
                                    size_t prof_id, int dircnt, bool terminate,
                                    RNG &rng) {
  double delta = m_mass_deltas(xfrm_id);
  if (dircnt == 0) {
    return;
  } else if (dircnt < 0) {
    dircnt = -dircnt;
    delta = -delta;
  }

  double parent_mass = m_tracker.find_mass_by_id(mass_id);
  size_t parent_id = mass_id;
  double new_mass = parent_mass;
  ProfileComponents &cur_prof = m_profiles[prof_id];

  // Apply the transformation `dir_cnt` times, breaking if we
  // don't meet mass or intensity thresholds
  while (dircnt > 0) {
    dircnt--;
    new_mass += delta;

    if (new_mass < m_mass_min || new_mass > m_mass_max) {
      m_stats_break_mass++;
      return;
    }

    double parent_intens = cur_prof.intensities[parent_id];
    double scale = (*m_iscale)(rng);
    double new_intensity = parent_intens * scale;
    if (new_intensity < m_min_intensity) {
      m_stats_break_intens++;
      return;
    }
    // Transfer intensity from parent to new
    auto [added_mass, added_id] =
        update_component(prof_id, new_mass, new_intensity);

    // If this metab is a dead end, mark it.
    if (m_enable_termination && terminate) {
      m_terminated.insert(added_id);
    }

    cur_prof.intensities[parent_id] *= (1 - scale);

    parent_mass = added_mass;
    parent_id = added_id;
    m_stats_applied[xfrm_id]++;
  }
}

}  // namespace massim
