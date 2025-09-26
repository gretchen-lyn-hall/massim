#include "fft_transformer.h"

namespace massim {

// class TransformList

TransformList::TransformList(ArrayRef mass_deltas, ArrayRef weights,
                             ArrayRef mean_lens, ArrayRef mean_lens_std,
                             bool same_len, bool always_center)
    : m_mass_deltas(mass_deltas), m_weights(weights), m_mean_lens(mean_lens),
      m_mean_lens_std(mean_lens_std), m_dist(weights.begin(), weights.end()),
      m_same_len(same_len), m_always_center(always_center)
      {
  if (mass_deltas.size() != weights.size() ||
      weights.size() != mean_lens.size() ||
      mean_lens.size() != mean_lens_std.size()) {
    throw std::invalid_argument("All input arrays must have same size.");
  }
}

std::vector<int> TransformList::pick_lens(double mean_len, size_t size,
                                          RNG &rng) const {
  // C++'s negative_binomial can't handle real-valued N
  // In general, one way to get a nb dist is:
  // X = rand_gamma(n, 1.0)
  // k = rand_poisson(X * (1-p)/p)
  std::vector<int> result;
  result.resize(size);

  // The input is the mean length of transform chains, assuming that the
  // transform is applied at least once. Thus the mean chain length is always
  // >=2
  // We use a neg binomial to model how many *additional* times to apply
  // the transform;
  std::gamma_distribution<> gamma_d(mean_len - 1.9999, 1.0);
  if (m_same_len) {
    double len =
        (1 + std::poisson_distribution<>(gamma_d(rng.engine))(rng.engine));
    for (int ii = 0; ii < size; ++ii) {
	result[ii] = len;
    }      
  } else {

    std::generate(result.begin(), result.end(), [&]() {
      return (1 + std::poisson_distribution<>(gamma_d(rng.engine))(rng.engine));
    });
  }    
  return result;
}

TransformSelection TransformList::choose_transform(const BoolMatType &presence,
                                                   RNG &rng) const {
  int x_id = m_dist(rng.engine);
  int num_vals = presence.count();
  auto counts = pick_lens(m_mean_lens[x_id], num_vals, rng);
  TransformSelection result;
  result.xfrm_id = x_id;
  result.delta_m = m_mass_deltas[x_id];
  result.forward_len.resize(counts.size());
  result.back_len.resize(counts.size());
  if (m_same_len) {

    int fwd;
    if (num_vals > 0)
	fwd = std::uniform_int_distribution<>(0, counts[0])(rng.engine);
    for (size_t ii = 0; ii < counts.size(); ii += 1) {
	  result.forward_len[ii] = fwd;
	  result.back_len[ii] = counts[ii] - fwd;
    }
  } else {
      int bias = std::uniform_int_distribution<>(0, 2)(rng.engine);
    if (m_always_center) {
	for (size_t ii = 0; ii < counts.size(); ii += 1) {
          int fwd = counts[ii] / 2;
          int rev = counts[ii] - fwd;
          if (bias == 1) {
            std::swap(fwd, rev);
	  }            
	  result.forward_len[ii] = fwd;
	  result.back_len[ii] = rev;
        }
        
    } else {      
      for (size_t ii = 0; ii < counts.size(); ii += 1) {
	  int fwd = std::uniform_int_distribution<>(0, counts[ii])(rng.engine);
	  result.forward_len[ii] = fwd;
	  result.back_len[ii] = counts[ii] - fwd;
      }
    }      
  }      

  return result;
}

// class TandemTransformer

TandemTransformer::TandemTransformer(ConstMatRef intensity_mat,
				     ConstArrayRef masses,
                                     const TransformList &transforms,
                                     const Distribution &intensity_scale,
                                     const Distribution &mass_dist,
                                     double mass_min, double mass_max,
                                     double min_intensity, double thresh_ppm)
    : m_intensity(intensity_mat), m_masses(masses), m_transforms(transforms),
      m_mass_min(mass_min), m_mass_max(mass_max),
      m_min_intensity(min_intensity), m_alignment_thresh(thresh_ppm * 1e-6),
      m_intensity_scale(std::move(intensity_scale.clone())),
      m_mass_dist(std::move(mass_dist.clone())) {

  if (intensity_mat.cols() != masses.size()) {
    throw std::invalid_argument(
        "Number of masses does not equal intensity columns.");
  }
}

TandemTransformer::TransformerResult
TandemTransformer::apply_transforms(RNG &rng, size_t max_new_peaks,
                                    size_t target_masses) {
  if (max_new_peaks == 0 && target_masses == 0) {
    throw std::invalid_argument(
        "One of 'max_new_peaks' or 'target_masses' must be nonzero");
  }

  TransformerResult result(*this, target_masses, max_new_peaks, rng);
  result.run(rng);
  return result;
}

// class TandemTransformer::TransformerResult

TandemTransformer::TransformerResult::TransformerResult(
    const TandemTransformer &parent, size_t target_masses, size_t max_new_peaks,
    RNG &rng)
    : m_parent(parent), m_targ_masses(target_masses),
      m_targ_peaks(max_new_peaks),
      m_num_peaks((m_parent.m_intensity > 0).count())

{
  // Initialize our sorted mass keys, as well as our indexed
  // intensity columns
  for (int ii = 0; ii < m_parent.m_intensity.cols(); ++ii) {
    m_mass_keys.insert(MassKey{m_parent.m_masses[ii], ii});
    m_i_columns.push_back(m_parent.m_intensity.col(ii));

    m_orig_id.push_back(ii);
    std::set<int> col_parent;
    col_parent.insert(ii);
    m_contrib.push_back(col_parent);
  }
}

MatType
TandemTransformer::TransformerResult::extend_matrix(ConstMatRef src_matrix,
                                                    bool use_contrib) const {
  if (src_matrix.cols() != m_parent.m_intensity.cols() ||
      src_matrix.rows() != m_parent.m_intensity.rows()) {
    throw std::domain_error("Shape of input matrix unlike parent matrix");
  }

  MatType result = MatType::Zero(num_rows(), m_mass_keys.size());
  size_t idx = 0;
  for (const auto &mass_it : m_mass_keys) {
    if (!use_contrib) {
      result.col(idx) = src_matrix.col(m_orig_id[mass_it.second]);
    } else {
      int cnt = 0;
      ArrayType accum = ArrayType::Zero(num_rows());
      for (auto orig : m_contrib[mass_it.second]) {
        accum += src_matrix.col(orig);
        cnt++;
      }
      result.col(idx) = accum / cnt;
    }
    idx++;
  }
  return result;
}

void TandemTransformer::TransformerResult::extend_chain(double start_mass,
                                                        size_t start_col_idx,
                                                        double mass_delta,
                                                        IntArrayType counter,
                                                        RNG &rng) {

  ArrayType intensity = m_i_columns[start_col_idx];
  BoolMatType presence = intensity > 0;
  double new_mass = start_mass;
  auto orig_id = m_orig_id[start_col_idx];
  auto composition = m_contrib[start_col_idx];

  int idx = 0;
  while ((counter >= 0).any()) {
    idx += 1;
    presence = counter > 0;
    if (!presence.any()) {
      break;
    }
    counter -= 1;
    new_mass += mass_delta;
    if (new_mass < m_parent.m_mass_min || new_mass > m_parent.m_mass_max)
      break;

    // For the destination of the transform, randomly scale the origin
    // intensity.
    for (int samp_id = 0; samp_id < presence.size(); ++samp_id) {
      if (presence.coeff(samp_id)) {
        intensity.coeffRef(samp_id) *= (*m_parent.m_intensity_scale)(rng);
      } else {
        intensity.coeffRef(samp_id) = 0;
      }
    }

    auto mass_it = find_mass(new_mass);
    // If the transformed mass is not within the threshold of any existing
    // mass, create a new column
    if (mass_it == m_mass_keys.end()) {
      // The new intensity column is just the scaled version of the
      // original intensity.
      // Note: we include intensities below the min_threshold (as later
      // transformations might push them above), but do not include
      // them in our count. However, at least one intensity value
      // must be above the thresh
      auto num_new = (intensity > m_parent.m_min_intensity).count();
      if (num_new == 0) {
        m_early_exit++;
        break;
      }
      ArrayType new_intensity = intensity;
      // Insert the new mass and column into our records
      size_t dest_idx = m_i_columns.size();
      m_i_columns.emplace_back(new_intensity);
      m_mass_keys.insert(MassKey{new_mass, dest_idx});
      m_num_peaks += num_new;
      // Gather stats about the "heritage" of the new column.
      m_orig_id.push_back(orig_id);
      m_contrib.push_back(composition);

    } else {
      // Otherwise update the existing mass column
      m_overlaps++;
      new_mass = mass_it->first;
      auto dest_idx = mass_it->second;
      auto &dest_intensity = m_i_columns[dest_idx];
      // Again, we allow the intensity to fall below the min threshold,
      // but don't count those as new peaks.
      m_num_peaks += (dest_intensity < m_parent.m_min_intensity &&
                      intensity > m_parent.m_min_intensity)
                         .count();
      dest_intensity += intensity;
      m_contrib[dest_idx].insert(start_col_idx);
    }
  }
}

void TandemTransformer::TransformerResult::run(RNG &rng) {
  while (true) {
    double targ_mass = (*m_parent.m_mass_dist)(rng);
    // Randomly pick an existing mass, and extract the intensity
    // information for it.

    // This part needs work. I want to select an existing mass, with
    // mid-range masses being more likely. This method kind of works;
    // it definitely picks masses towards the center (assuming the input
    // is a normal dist). However, it is biased; if there are two very
    // closely spaced masses, it will almost never select the larger of
    // the two
    auto mass_it = m_mass_keys.lower_bound(MassKey{targ_mass, 0});
    if (mass_it == m_mass_keys.end())
      mass_it--;
    double start_mass = mass_it->first;
    assert(start_mass > m_parent.m_mass_min &&
           start_mass < m_parent.m_mass_max);
    double start_col = mass_it->second;
    BoolMatType presence = m_i_columns[start_col] > 0;

    // Determine which transform to apply, and how many times to apply
    // it
    auto xfrm_dat = m_parent.m_transforms.choose_transform(presence, rng);

    // Apply the transform in either direction

    auto a1 = presence.count();
    extend_chain(start_mass, start_col, xfrm_dat.delta_m,
                 fill_from_mask(presence, xfrm_dat.forward_len), rng);

    assert(a1 == presence.count());
    extend_chain(start_mass, start_col, -xfrm_dat.delta_m,
                 fill_from_mask(presence, xfrm_dat.back_len), rng);

    // Quit if we've reached any target (should also add a timeout?)
    if (m_targ_peaks != 0 && m_num_peaks > m_targ_peaks)
      break;
    if (m_targ_masses != 0 && m_mass_keys.size() >= m_targ_masses)
      break;
  }
}
} // namespace massim
