#include "fft_transformer.h"
#include <ranges>
#include <cassert>

namespace massim {

// class TransformList

TransformList::TransformList(ConstArrayRef mass_deltas, ConstArrayRef weights,
                             ConstArrayRef mean_lens, ConstArrayRef mean_lens_std,
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

TransformSelection TransformList::choose_transform(RNG &rng) const {
  // Pick a random transformation based on frequency
  int x_id = m_dist(rng.engine);
  // Pick the total chain length for each non-zero metabolite  
  auto counts = pick_lens(m_mean_lens[x_id], 1, rng);
  TransformSelection result;

  // The rest of this is just divvying up the chain lengths between forward
  // and reverse transformations, depending on the settings.  
  result.xfrm_id = x_id;
  result.delta_m = m_mass_deltas[x_id];
  result.forward_len.resize(1);
  result.back_len.resize(1);
  int fwd = std::uniform_int_distribution<>(0, counts[0])(rng.engine);
  result.forward_len[0] = fwd;
  result.back_len[0] = counts[0] - fwd;
  return result;
}

TransformSelection TransformList::choose_transform(const BoolMatType &presence,
                                                   RNG &rng) const {
  // Pick a random transformation based on frequency
  int x_id = m_dist(rng.engine);
  int num_vals = presence.count();
  // Pick the total chain length for each non-zero metabolite  
  auto counts = pick_lens(m_mean_lens[x_id], num_vals, rng);
  TransformSelection result;

  // The rest of this is just divvying up the chain lengths between forward
  // and reverse transformations, depending on the settings.  
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

TandemTransformer::TandemTransformer(
    ConstMatRef intensity_mat, ConstArrayRef masses,
    const TransformList &transforms, const Distribution &intensity_scale,
    const Distribution &mass_dist, double mass_min, double mass_max,
    double min_intensity, double thresh_ppm, PickMassMode pick_mode,
    double mass_center, double mass_scale
    )
    : m_intensity(intensity_mat), m_masses(masses), m_transforms(transforms),
      m_mass_min(mass_min), m_mass_max(mass_max),
      m_min_intensity(min_intensity), m_alignment_thresh(thresh_ppm * 1e-6),
      m_intensity_scale(std::move(intensity_scale.clone())),
      m_mass_dist(std::move(mass_dist.clone())), m_pick_mode(pick_mode),
      m_freq_ctr(mass_center), m_freq_scale(mass_scale)
      {
    
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
      m_num_peaks((m_parent.m_intensity > 0).count()),
      m_weighted_peaks(0.0),
      m_freq_dist(0, 1)
{
  // To keep things efficient, we store intensity columns in an unsorted
  // vector that we can easily append to. In a parallel (sorted) map, we store
  // the mass of each column, and its index in the vector. Here, we initialize
  // our sorted mass keys, as well as the initial 
  // intensity columns
  for (size_t ii = 0; ii < m_parent.m_intensity.cols(); ++ii) {
    int non_zero = (m_parent.m_intensity.col(ii) > 0).count();
    double weighted = non_zero * weight_peak(m_parent.m_masses[ii]);

    m_mass_keys.insert(std::make_pair(
			   m_parent.m_masses[ii],
			   MassInfo{ii, non_zero, weighted}));
    m_i_columns.push_back(m_parent.m_intensity.col(ii));

    // Weighted peaks are used for frequency-dependent choice of metabolites
    m_weighted_peaks += weighted; 

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
	result.col(idx) = src_matrix.col(m_orig_id[mass_it.second.column_id]);
    } else {
      int cnt = 0;
      ArrayType accum = ArrayType::Zero(num_rows());
      for (auto orig : m_contrib[mass_it.second.column_id]) {
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

  // This is the method that does the work of taking a source mass, and
  // adding transformations in a single direction
  // Given a starting mass/intensity column, it takes the transformation
  // mass and a list of chain lengths and applies the transformation

  // Here, we *copy* the parent intensity.  
  ArrayType intensity = m_i_columns[start_col_idx];
  // There is a bit of redundancy here - `counter` contains an entry for
  // every non-zero metabolite, which we computed previously.   
  BoolMatType presence = intensity > 0;
  double new_mass = start_mass;
  auto orig_id = m_orig_id[start_col_idx];
  auto composition = m_contrib[start_col_idx];

  int idx = 0;
  while ((counter >= 0).any()) {
      // Keep going until we've reached the max chain length (counter all zero)
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

    MassIter mass_it = find_mass(new_mass);
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
      m_mass_keys.insert(MassKey{
          new_mass,
          {dest_idx, static_cast<int>(num_new), num_new * weight_peak(new_mass)}
          });
      m_num_peaks += num_new;
      m_weighted_peaks +=  num_new * weight_peak(new_mass);
      // Gather stats about the "heritage" of the new column.
      m_orig_id.push_back(orig_id);
      m_contrib.push_back(composition);

    } else {
      // Otherwise update the existing mass column by adding to it
      m_overlaps++;
      new_mass = mass_it->first;
      auto dest_idx = mass_it->second.column_id;
      auto &dest_intensity = m_i_columns[dest_idx];
      // Again, we allow the intensity to fall below the min threshold,
      // but don't count those as new peaks.
      int non_zero = (dest_intensity < m_parent.m_min_intensity &&
                      intensity > m_parent.m_min_intensity)
                         .count();
      double weight = non_zero * weight_peak(new_mass);
      m_num_peaks -= mass_it->second.count;
      m_weighted_peaks -= mass_it->second.weighted;
      m_num_peaks += non_zero;
      m_weighted_peaks += weight;
      mass_it->second.count = non_zero;
      mass_it->second.weighted = weight;
      dest_intensity += intensity;
      m_contrib[dest_idx].insert(start_col_idx);
    }
  }
}

TandemTransformer::TransformerResult::MassIter
TandemTransformer::TransformerResult::pick_mass(RNG &rng) {
  if (m_parent.m_pick_mode == PICK_BY_MASS) {
    double targ_mass = (*m_parent.m_mass_dist)(rng);
    // Randomly pick an existing mass, and extract the intensity
    // information for it.

    // This part needs work. I want to select an existing mass, with
    // mid-range masses being more likely. This method kind of works;
    // it definitely picks masses towards the center (assuming the input
    // is a normal dist). However, it is biased; if there are two very
    // closely spaced masses, it will almost never select the larger of
    // the two
    auto mass_it = m_mass_keys.lower_bound(targ_mass);
    if (mass_it == m_mass_keys.end())
      mass_it--;
    return mass_it;
  } else if (m_parent.m_pick_mode == PICK_BY_FREQ) {
    // Here, we roll our own version of std::discrete_distribution
    // The sum of all weighted mass counts should equal m_weighted_peaks, so
    // pick a number between 0 and m_weighted_peaks, and iterate through masses,
    // summing weights, until we exceed the target.

    // Note, this is currently O(N) in number of masses, and seems to slow
    // things down.
    // We can make this O(log(N)), but we need a balanced tree implementation
    // where we can compute partial sums.    
    double target_val = m_weighted_peaks * m_freq_dist(rng);
    double accum = 0.0;
    auto it = m_mass_keys.begin();
    while (it != m_mass_keys.end()) {
      accum += it->second.weighted;
      if (accum >= target_val)
        return it;
      it++;
    }
    assert(it != m_mass_keys.end());
    
  } else {
      throw std::domain_error("Unknown enum value for pick mass"); 
  }    
  
}



void TandemTransformer::TransformerResult::run(RNG &rng) {
  while (true) {
    auto mass_it = pick_mass(rng);
    double start_mass = mass_it->first;
    assert(start_mass > m_parent.m_mass_min &&
           start_mass < m_parent.m_mass_max);
    size_t start_col = mass_it->second.column_id;
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
