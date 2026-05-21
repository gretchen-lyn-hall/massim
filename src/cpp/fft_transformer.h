#ifndef MASSIM_FFT_TRANSFORMER_H
#define MASSIM_FFT_TRANSFORMER_H

#include "common.h"
#include "distribution.h"

#include <random>
#include <set>
#include <cassert>
namespace massim {


enum PickMassMode {
  PICK_BY_MASS = 0,
  PICK_BY_FREQ = 1,
  PICK_BY_WEIGHT = 2
};  

struct TransformSelection {
  int xfrm_id;
  double delta_m;
  IntArrayType forward_len;
  IntArrayType back_len;
};

class TransformList {
  ArrayType m_mass_deltas;
  ArrayType m_weights;
  ArrayType m_mean_lens;
  ArrayType m_mean_lens_std;
  mutable std::discrete_distribution<int> m_dist;
    bool m_same_len;  
    bool m_always_center;
public:
  // TransformList contains the profile of what biochemical transformations to
  // apply, and how to apply them.
  //  'mass_deltas': For each transformation, its (positive) mass difference
  //  'weights': Weighting for determining which transform to pick. Higher
  //             weight = higher probability.
  //  'mean_lens': Mean length of transform chain (how many times in a row to
  //               apply it), assuming that the transform has already been
  //               picked.  Must be >=2 for each
  //               transform
  //  'mean_lens_std': Deviation of mean lens. Reserved for future use.
  // All arrays must have the same length.
  TransformList(ConstArrayRef mass_deltas, ConstArrayRef weights, ConstArrayRef mean_lens,
                ConstArrayRef mean_lens_std, bool same_len, bool always_center);

  std::vector<int> pick_lens(double mean_len, size_t size, RNG &rng) const;

    // Choose a transformation for a single metabolite
  TransformSelection choose_transform(RNG &rng) const;
    // Choose a transformation for a matrix of metabolites
  TransformSelection choose_transform(const BoolMatType &presence,
                                      RNG &rng) const;

  size_t num_xfrms() const {
    return m_mass_deltas.size();
  }    
};

class TransformTracker {
  typedef std::set<double>::iterator mass_iter_t;
  
  std::set<double> m_masses;
  std::vector<double> m_mass_deltas;
  IntArrayType m_counts;
  double m_min_mass;
    double m_max_mass;
    double m_tol;
    size_t m_total;
public:
  TransformTracker(ConstArrayRef mass_deltas, double tol_ppm)
      : m_mass_deltas(mass_deltas.begin(), mass_deltas.end()),
        m_counts(IntArrayType::Zero(mass_deltas.size())), m_tol(tol_ppm * 1e-6),
        m_min_mass(std::numeric_limits<double>::infinity()),
        m_max_mass(-std::numeric_limits<double>::infinity())
        {
  }


  bool add_mass(double mass) {
    if (contains(mass))
      return false;

    for (int ii = 0; ii < m_mass_deltas.size(); ++ii) {
	double delta = m_mass_deltas[ii];
        if (contains(mass - delta)) {
          m_counts(ii) += 1;
	  m_total += 1;
	}          
        if (contains_2(mass + delta, mass)) {
          m_counts(ii) += 1;
	  m_total += 1;
	}          
    }
    m_masses.insert(mass);
    m_min_mass = std::min(m_min_mass, mass);
    m_max_mass = std::max(m_max_mass, mass);
    return true;
  }

  const IntArrayType& counts() const {
      return m_counts;
  }

  size_t total_count() const { return m_total; }
  
  ArrayType masses() const {
    ArrayType result(m_masses.size());
    size_t idx = 0;
    for (const double x : m_masses) {
      result(idx) = x;
      idx++;
    }      
    return result;
  }

  bool contains(double test_mass) const {
    if (test_mass < m_min_mass * (1 - m_tol) ||
        test_mass > m_max_mass * (1 + m_tol))
	return false;

    auto it = m_masses.lower_bound(test_mass);
    if (it != m_masses.end()) {
        if (abs(*it - test_mass) / *it < m_tol)
          return true;
    }        
    if (it != m_masses.begin()) {
	it--;
	assert((test_mass - *it) >= 0);
        if (abs(*it - test_mass) / *it < m_tol)
          return true;
    }
    return false;      
  }

  bool contains_2(double test_mass, double denom_mass) const {
    double tol_abs = denom_mass * m_tol;
    if (test_mass < m_min_mass - tol_abs ||
        test_mass > m_max_mass  + tol_abs)
	return false;

    auto it = m_masses.lower_bound(test_mass);
    if (it != m_masses.end()) {
        if (abs(*it - test_mass) < tol_abs)
          return true;
    }        
    if (it != m_masses.begin()) {
	it--;
	assert((test_mass - *it) >= 0);
        if (abs(*it - test_mass) < tol_abs)
          return true;
    }
    return false;      
  }    
  
private:
};

class TransformWeighter : public TransformTracker {
  ArrayType m_probs;
  double m_preweight;
  ArrayType m_pre_probs;
    
public:
  TransformWeighter(ConstArrayRef mass_deltas, ConstArrayRef probs,
                    double preweight, double tol_ppm)
      : TransformTracker(mass_deltas, tol_ppm), m_probs(probs), m_preweight(preweight) {
    m_probs /= m_probs.sum();
    m_pre_probs = m_probs * m_preweight;
    }

    ArrayType weights() const { return m_pre_probs + counts().cast<double>(); }

    int choose(RNG &rng) const {
	
    }      
    
};



struct MassInfo {
    size_t column_id;
    int count;
    double weighted;
};

typedef std::map<double, MassInfo> MassMap;

class TandemTransformer {

  typedef std::pair<double, MassInfo> MassKey;

  MatType m_intensity;
  ArrayType m_masses;
  TransformList m_transforms;
  double m_mass_min;
  double m_mass_max;
  double m_min_intensity;
  double m_alignment_thresh;
  std::unique_ptr<Distribution> m_intensity_scale;
  std::unique_ptr<Distribution> m_mass_dist;
  // m_mass keys stores the masses in sorted order, along with the index
  // of the associated intensity column (initially in order, but as new
  // columns are added, they will be added out of order).
  MassMap m_mass_keys;
  // Intensity for each mass
  std::vector<ArrayType> m_i_columns;
    PickMassMode m_pick_mode;
    double m_freq_ctr;
    double m_freq_scale;

  public:
    // Creates a class that adds new biochemical transformations to a
    // dataset.
    // `intensity_mat`: A compound x sample matrix of intensity values.
    // `masses`: An array of compound masses. Must have same length as
    //   `intensity_mat`
    // `transforms`: An instance of TransformList, used to select which
    //    transforms to apply.
    // `intensity_scale`: A Distribution that determines the scaling factor used
    //     when creating new compounds. The new compound's intensity is the
    //     parent compounds intensity times this scale.
    // `mass_dist`: Used for picking compounds to transform; see `mass_mode`
    //
    // `mass_min` & `mass_max`: Range of valid compound masses.
    // `min_intensity`: Peaks below this intensity will not be added.
    // `thresh_ppm`: minimum distance between peaks in parts per million;
    //    peaks closer than this will be merged.
    // `mass_mode`: How we pick compounds to transform.
    //       BY_MASS: randomly select a mass using `mass_dist`, and choose
    //                closest peak to it,
    //       BY_FREQ: Peaks are chosen by frequency; the probability of picking
    //                a peak is proportional to the number of samples it appears
    //                in. This can also be weighted by mass; see next two args
    // `mass_center`: When BY_FREQ is specified, weights the probability of
    //                a peak not only by the number of samples it appears in,
    //                but by proximity to this center. The weight is given by
    //                10**-((m-mass_center)/mass_scale)**2
    // 
//    
  TandemTransformer(ConstMatRef intensity_mat, ConstArrayRef masses,
                    const TransformList &transforms,
                    const Distribution &intensity_scale,
                    const Distribution &mass_dist, double mass_min = 150,
                    double mass_max = 1200, double min_intensity = 1,
                    double thresh_ppm = 1.0,
                    PickMassMode mass_mode = PICK_BY_MASS,
                    double mass_center = 600,
		    double mass_scale = 100
                    );

  class TransformerResult {
    // Class for computing and returning a transformation simulation.
    const TandemTransformer &m_parent;
    size_t m_targ_masses;
    size_t m_targ_peaks;
    size_t m_num_peaks;
      double m_weighted_peaks;
    // m_mass keys stores the masses in sorted order, along with the index
    // of the associated intensity column (initially in order, but as new
    // columns are added, they will be added out of order).
    MassMap m_mass_keys;
    // Intensity for each mass
    std::vector<ArrayType> m_i_columns;
    // The id of the mass in the original dataset that directly led
    // to the creation of this mass
    std::vector<int> m_orig_id;
    // The id of all masses in the original dataset that contributed to
    // this mass
    std::vector<std::set<int>> m_contrib;
    // Stats:
    size_t m_early_exit;
    size_t m_overlaps;
    UniformDistribution m_freq_dist;
    typedef MassMap::iterator MassIter;
    typedef MassMap::const_iterator MassIterConst;

  public:
    TransformerResult(const TandemTransformer &parent, size_t target_masses,
                      size_t max_new_peaks, RNG &rng);

    size_t num_rows() const { return m_parent.m_intensity.rows(); }

    double weight_peak(double mass) const {
	// Gaussian curve centered at m_freq_ctr
	auto ctr =( mass - m_parent.m_freq_ctr) / m_parent.m_freq_scale;
        return pow(10, -(ctr * ctr));
    }


      // Return the intensity as a dense matrix
    MatType intensity() const {
      MatType result = MatType::Zero(num_rows(), m_mass_keys.size());
      size_t idx = 0;
      for (const auto &mass_it : m_mass_keys) {
        result.col(idx) = m_i_columns[mass_it.second.column_id];
        idx++;
      }
      return result;
    }

      // Return the intensity as a sparse matrix
      Eigen::SparseMatrix<double> sparse_intensity() const {
	  Eigen::SparseMatrix<double> result(num_rows(), m_mass_keys.size());
          std::vector<Eigen::Triplet<double>> trips;
	  trips.reserve(m_num_peaks);          
	for (int mass_idx = 0; mass_idx < m_mass_keys.size(); ++mass_idx) {
          const auto &mass_it = *std::next(m_mass_keys.begin(), mass_idx);
          const auto &col = m_i_columns[mass_it.second.column_id];
          for (int row_idx = 0; row_idx < col.size(); ++row_idx) {
	      if (col[row_idx] > 0) {
		  trips.emplace_back(row_idx, mass_idx, col[row_idx]);
	      }
          }
	}          
	result.setFromTriplets(trips.begin(), trips.end());
	return result;
    }


    // For each peak, returns the index of `ancestor peak` in the original data
    // which was transformed to create it (or its original index, if it existed
    // originally.    
    IntArrayType original_ids() const {
      IntArrayType result(m_mass_keys.size());
      int idx = 0;
      for (const auto m_it : m_mass_keys) {
        result[idx] = m_orig_id[m_it.second.column_id];
        ++idx;
      }
      return result;
    }

      // Return an array of masses for all peaks.    
      ArrayType masses() const {
	  ArrayType result(m_mass_keys.size());
	  std::transform(m_mass_keys.begin(),
		  m_mass_keys.end(),
		  result.begin(),
		  [](const auto& mkey){return mkey.first;});
	  return result;
      }

    // Given a matrix of the same shape as the intensity matrix used for
    // the simulation, 'extend_matrix' essentially copies the steps used
    // during the transformation simulation to create an output matrix of
    // the same shape as the output intensity matrix. It is intended to be
    // used to maintain auxiliary matrices (like probability) to go along
    // with the intensity matrix.
    // In the simulation step, each new column in the output intensity matrix
    // is built from one or more input columns. Thus, each output column
    // has a "parent" (the original input column used to create the output)
    // and "contributors" (other input columns that added to the intensity
    // after it was created).
    // `extend_matrix` uses that history to extend other matrices.
    // In the simplest case (use_contrib=false), each output column is a copy
    // of the parent column in the input. If use_contrib=true, then each output
    // column is the average of all contributing columns in the input.
    MatType extend_matrix(ConstMatRef src_matrix, bool use_contrib = false) const;

  private:
      MassIter pick_mass(RNG &rng);
    
    bool mass_match(double m1, double m2) const {
      return abs(m1 - m2) / m1 < m_parent.m_alignment_thresh;
    }

    // Given a mass, search existing masses to see if any are within
    // the threshold.
    // If a mass is found, return an iterator to the mass within mass_keys.
    // Otherwise, mass_keys.end() is returned.
    MassIter find_mass(double m1) {
      auto it = m_mass_keys.lower_bound(m1);
      if (it != m_mass_keys.end() && mass_match(it->first, m1))
        return it;
      if (it != m_mass_keys.begin()) {
        it--;
        if (mass_match(it->first, m1))
          return it;
      }
      return m_mass_keys.end();
    }

    // Implementation of a type of NumPy logical indexing, where if
    // you have a mask of length N, with k true values, you can
    // write:
    //   A[mask] = data
    // when A is of length N and 'data' has k values.
    // This is different I believe than Eigen::Select, which only works
    // when 'data' is of size N.
    IntArrayType fill_from_mask(const BoolMatType mask,
                                const IntArrayType &data) {
      IntArrayType result = IntArrayType::Zero(mask.size());
      int dat_idx = 0;
      for (int ii = 0; ii < mask.size(); ++ii) {
        if (mask.coeff(ii)) {

          assert(dat_idx < data.size());

          result.coeffRef(ii) = data.coeff(dat_idx);
          dat_idx += 1;
        }
      }
      assert(dat_idx == data.size());
      return result;
    }

    // Apply the transform (given by mass delta) to the columns in a single
    // direction. The origin of the transform is given by start_col_idx.
    // For each row (i.e. sample) the number of times to apply it is stored
    // in counter.
    void extend_chain(double start_mass, size_t start_col_idx,
                      double mass_delta, IntArrayType counter, RNG &rng);

  protected:
    void run(RNG &rng);
    friend class TandemTransformer;
  };

  TransformerResult apply_transforms(RNG &rng, size_t max_new_peaks = 0,
                                     size_t target_masses = 0);
};



struct ProfileComponent {
    double weight;
    double prob_mass;
};


class ProfileGenerator {

  // We have two parallel data structures here:
  //  - MassMap: For each peak, stores a map keyed by profile_id that contains
  //      the weight for that profile
  //    This is where most of the work happens
  //  - ProfileItems: For each profile, the mass and probability of all peaks
  //    in the profile. This is just used for picking the next mass for a
  //    given profile, and the ordering is unimportant.
  
  
    struct ProfileReference {
	double weight;
    };
    // Map of profile id to profile component
    typedef std::map<size_t, ProfileReference> ComponentMap;
    struct MassEntry {
	ComponentMap prof_comps;
    };

    typedef std::map<const double, MassEntry> MassMap;
    typedef MassMap::iterator MassIter;
    typedef ComponentMap::iterator CompIter;


    struct ProfileEntry {
	double mass;
	double prob_mass;
    };

    struct ProfileRecord {
      std::vector<ProfileEntry> entries;
      double prob_total;
      double weight_total;
	int profile_id;

      const size_t size() const { return entries.size(); }
	const std::vector<ProfileEntry>::const_iterator begin() const {return entries.begin();}
	const std::vector<ProfileEntry>::const_iterator end() const {return entries.end();}
      const ProfileEntry &operator[](size_t ii) { return entries[ii]; }
	void add_mass(double m, double w) {entries.push_back({m, w}); prob_total += w;}	
    };      
    
   
    std::vector<ProfileRecord> m_profiles;
    MassMap m_masses;
  TransformList m_transforms;
  double m_mass_min;
  double m_mass_max;
  double m_min_intensity;
  double m_alignment_thresh;
  std::unique_ptr<Distribution> m_intensity_scale;
  std::unique_ptr<Distribution> m_mass_dist;
    PickMassMode m_pick_mode;
    double m_freq_ctr;
    double m_freq_scale;
    int m_num_components;
    int m_break_mass;
    int m_break_intensity;
    int m_break_len;    

  public:
    struct ProfileResult {
      IntArrayType indices;
	ArrayType weights;
   };

    
    // Creates a class that adds new biochemical transformations to a
    // dataset.
    // `intensity_mat`: A compound x species matrix of profile weights
    //                  for the core metabolites
    // `masses`: An array of compound masses. Must have same length as
    //   `intensity_mat`
    // `transforms`: An instance of TransformList, used to select which
    //    transforms to apply.
    // `intensity_scale`: A Distribution that determines the scaling factor used
    //     when creating new compounds. The new compound's intensity is the
    //     parent compounds intensity times this scale.
    // `mass_dist`: Used for picking compounds to transform; see `mass_mode`
    //
    // `mass_min` & `mass_max`: Range of valid compound masses.
    // `min_intensity`: Peaks below this intensity will not be added.
    // `thresh_ppm`: minimum distance between peaks in parts per million;
    //    peaks closer than this will be merged.
    // `mass_mode`: How we pick the components of a profile to transform
    //       BY_MASS: Pick using a gaussian on the masses.
    //       BY_FREQ: Currently, completely random choice of components.
    // `mass_center`: When BY_FREQ is specified, weights the probability of
    //                a peak not only by the number of samples it appears in,
    //                but by proximity to this center. The weight is given by
    //                10**-((m-mass_center)/mass_scale)**2
    // 
//    
  ProfileGenerator(int num_profiles, 
                    const TransformList &transforms,
                    const Distribution &intensity_scale,
                    double mass_min = 150,
                    double mass_max = 1200, double min_intensity = 1e-6,
                    double thresh_ppm = 1.0,
                    PickMassMode mass_mode = PICK_BY_MASS,
                    double mass_center = 600,
		    double mass_scale = 100
                    );

  size_t num_profiles() const { return m_profiles.size(); }
  size_t num_masses() const { return m_masses.size(); }
  size_t num_components() const {
    size_t result = 0;
    for (const auto& prof : m_profiles) {
      result += prof.size();
    }
    return result;
  }

// Essentially the PDF for a normal distribution of masses
    double weight_peak(double mass) const {
	// Gaussian curve centered at m_freq_ctr
	auto ctr =( mass - m_freq_ctr) / m_freq_scale;
        return pow(10, -(ctr * ctr));
    }


      // Return an array of masses for all peaks.    
      ArrayType masses() const {
	  ArrayType result(m_masses.size());
	  std::transform(m_masses.begin(),
		  m_masses.end(),
		  result.begin(),
		  [](const auto& mkey){return mkey.first;});
	  return result;
      }

      // Return the current profiles. Each contains indices into
// the list of masses, and an equally sized array of weights.      
      std::vector<ProfileResult> profiles() const {
        std::vector<ProfileResult> result;
	std::vector<size_t> cur_indices;
	// Prepare the result vectors and arrays
        result.resize(m_profiles.size());
        cur_indices.resize(m_profiles.size()); // Init to 0s

        for (std::size_t ii=0; ii < m_profiles.size(); ++ii) {
	    result[ii].indices.resize(m_profiles[ii].size());
	    result[ii].weights.resize(m_profiles[ii].size());
        }

	int mass_idx = 0;
        for (const auto &mass_it : m_masses) {
          for (const auto &prof_id : mass_it.second.prof_comps) {
	    auto &profile = result[prof_id.first];
	    size_t& cur_idx = cur_indices[prof_id.first];
            profile.indices[cur_idx] = mass_idx;
            profile.weights[cur_idx] = prof_id.second.weight;
	    cur_idx++;
          }
	  mass_idx++;
	}          
	  return result;
      }


      // Find the component for the given mass and profile, creating an
    // empty component if necessary.      
      CompIter add_component(size_t profile_id, double mass) {
        if (profile_id >= m_profiles.size()) {
          throw std::runtime_error("Profile ID out of bounds");
        }
        auto m_it = find_mass(mass);
        if (m_it == m_masses.end()) {
          // If it's new, create the mass, create the component, and update
	    // the profile record.
          m_it = m_masses.insert({mass, {}}).first;
          auto p_it = m_it->second.prof_comps.insert({profile_id, {}}).first;
          m_profiles[profile_id].add_mass(m_it->first, weight_peak(m_it->first));
	  m_num_components++;
	  return p_it;
        } else {
	    // Get list of components for existing mass
	    auto &mass_map = m_it->second.prof_comps;
            auto [p_it, inserted] = mass_map.insert({profile_id, {}});
            if (inserted) {
              // If we created the component, we have to update
              // the profile record
		m_profiles[profile_id].add_mass(m_it->first, weight_peak(m_it->first));
	      m_num_components++;
            }
	    return p_it;

	}          
	
      }

     // Add a component of a given weight to profile.
      // If overwrite is `false` and any existing component is nonzero,
      // the component will not be changed and `false` will be returned.
      // In all other cases, the component is written and `true` returned.      
    bool set_component(size_t profile_id, double mass, double weight, bool overwrite=true) {
        auto p_it = add_component(profile_id, mass);
        if (not overwrite and p_it->second.weight != 0)
	    return false;
        p_it->second.weight = weight;
	return true;        
    }

    // Randomly apply transforms to randomly selected profiles until
    // either the target masses or components are met. If neither specified,
// add a single new component.
    void apply_transforms(RNG &rng, size_t targ_components = 0,
                          size_t targ_masses = 0);

    std::map<std::string, double> stats() const {
      std::map<std::string, double> result;
      result["break_mass"] = m_break_mass;
      result["break_intensity"] = m_break_intensity;
      result["break_total"] = m_break_len;
      return result;
    }      

  private:
    // Pick a random mass component from the given profile.
    MassIter pick_component(ProfileRecord &profile, RNG &rng);

    void recompute_weights() {
	for ( auto & prof : m_profiles) {
          prof.weight_total = 0;
        }
        for (auto &mass_it : m_masses) {
          for (auto &comp_it : mass_it.second.prof_comps) {
	      m_profiles[comp_it.first].weight_total += comp_it.second.weight;
          }
	}         
    }      
    
    bool mass_match(double m1, double m2) const {
      return abs(m1 - m2) / m1 < m_alignment_thresh;
    }

    // Given a mass, search existing masses to see if any are within
    // the threshold.
    // If a mass is found, return an iterator to the mass within mass_keys.
    // Otherwise, mass_keys.end() is returned.
    MassIter find_mass(double m1) {
      auto it = m_masses.lower_bound(m1);
      if (it != m_masses.end() && mass_match(it->first, m1))
        return it;
      if (it != m_masses.begin()) {
        it--;
        if (mass_match(it->first, m1))
          return it;
      }
      return m_masses.end();
    }


    // Apply the transform (given by mass delta) to a given profile, 
      void extend_chain(MassIter start_component,
			double mass_delta,
			size_t profile_id, 
			int counter,
			RNG &rng);

};


} // namespace massim
#endif // MASSIM_FFT_TRANSFORMER_H
