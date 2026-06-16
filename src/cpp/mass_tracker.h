#ifndef MASSIM_MASS_TRACKER_H
#define MASSIM_MASS_TRACKER_H

#include <cassert>
#include "common.h"
#include <vector>
// #include <flat_map>
#include "flat_map.h"
#include <unordered_set>


namespace massim {


enum class TransformMassMode { MODE_STRICT = 0, MODE_MODERATE = 1, MODE_LAX = 2 };

// Source - https://stackoverflow.com/a/9729747
// Posted by Kerrek SB, modified by community. See post 'Timeline' for change history
// Retrieved 2026-05-26, License - CC BY-SA 3.0

template <class T>
void hash_combine(size_t& seed, T const& v)
{
  // from https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
  if constexpr (sizeof(size_t) >= 8u)
  {
    seed ^= v + 0x517cc1b727220a95 + (seed << 6u) + (seed >> 2u);
  }
  else
  {
    seed ^= v + 0x9e3779b9 + (seed << 6u) + (seed >> 2u);
  }
}


template <typename T>
struct pair_hash{};

template <typename S, typename T>
struct pair_hash<std::pair<S, T>>
{
    inline std::size_t operator()(const std::pair<S, T> & v) const
    {
         std::size_t seed = 0;
         hash_combine(seed, std::hash<S>{}(v.first));
         hash_combine(seed, std::hash<T>{}(v.second));
         return seed;
    }
};


    

class MassTracker {
  // Continuously track occurrences of particular mass differences (i.e.
  // "transformations") occurring in an updating list of masses.

    stdext::flat_map<double, size_t> m_masses;
    // This is where using a boost::bimap would be useful!
    std::unordered_map<size_t, double> m_mass_ids;
    TransformMassMode m_mode;
    bool m_strict;
    bool m_track_applications;
    std::vector<double> m_mass_deltas;
    std::unordered_set<std::pair<size_t, int>,
                       pair_hash<std::pair<size_t, int>>> m_seen;
    IntArrayType m_counts;
    std::vector<std::tuple<size_t, size_t, size_t>> m_applications;
    double m_min_mass {std::numeric_limits<double>::infinity()};
  double m_max_mass {-std::numeric_limits<double>::infinity()};
    double m_tol;
  size_t m_total {};

public:
  using mass_iter_t = stdext::flat_map<double, size_t>::iterator;
  using const_mass_iter_t = stdext::flat_map<double, size_t>::const_iterator;
    // Type for tracking individial transforms, of type <xfrm_id, src_id, dst_id>
    using applications_t = std::vector<std::tuple<size_t, size_t, size_t>>;
  
  // Create a tracker that will continuously update the counts of occurrences of
  // specified mass deltas in a list of masses.
  // For synthesizing data, this can be used to weight the probability of
  // transformations to apply. Or, it can be used to count transformations in an
  // existing data set.
  // This approach can be orders of magnitude faster than finding and scanning
  // all mass differences. That approach is is of order O(N^2 * logN * logM),
  // where N is the number of masses, and M the number of mass deltas.
  // By tracking, the order is O(N * logN * M) (N insertions, M comparisons with
  // logN lookup for each).
  //
  // In the following, the notations "source" and "dest(ination)" are used
  // to refer to the masses involved in a transformation / mass delta.
  // "Source" always refers to the lower mass, and "dest" to the higher.
  // `m_src`, `m_dst` are source and dest masses such that
  // `m_src + D ~= m_dst`.



  // Arguments:
  // - mass_deltas: list of (positive) mass deltas to look for.
  // - tol_ppm: The relative tolerance with which to decide if the difference
  //            between two masses in the set constitutes a transformation,
  // - mode: Mode for tolerance. This mode sets the
  //         criteria for when `(m_dst - m_src)` is considered close to `D`.
  //         That is, if `err = abs((m_dst - m_src) - D)`:
  //         - MODE_STRICT: `err` is within `tol_ppm` of `m_lo`
  //         - MODE_LAX: `err` is within `tol_ppm` of `m_hi`
  //         - MODE_MODERATE: `err` is within `tol_ppm` of `sqrt(m_lo*m_hi)`
  // - strict_count: Due to the tolerance, it is possible that a particular
  //         mass delta `D` is counted twice for the same source (low mass),
  //         or the same destination (high mass). That is, for a given mass
  //         `m_src`, `m_src + D` might be within tolerance of more than one
  //         tracked mass.
  //         (or alternatively, `m_dst - D` may match more than one tracked
  //         mass as a source).
  //         If `strict_count` is false, all possible transforms are counted,
  //         whereas if it is true, a given transform `D` will be counted at
  //         most once as a source and once as a destination for any given
  //         mass.
  //         Note: if you are tracking applications of each transforms (see
  //         below), the choice of which particular source and destination
  //         mass is arbitrary. In that case, it may be better to set
  //         `strict_count` to false, track all applications, and clean up the
  //         data in a post-processing step (e.g. preserve the applications
  //         with the least error)
  // - track_applications: If true, MassTracker will record the source and
  //         destination for every detected transformation, which can be
  //         retrieved with `applications()`. The masses are identified by
  //         the order they were inserted into the tracker.
    MassTracker(ConstArrayRef mass_deltas, double tol_ppm, TransformMassMode mode,
              bool strict_count, bool track_applications)
      : m_mass_deltas(mass_deltas.begin(), mass_deltas.end()), m_mode(mode),
        m_counts(IntArrayType::Zero(mass_deltas.size())), m_tol(tol_ppm * 1e-6),
        m_strict(strict_count),
	m_track_applications(track_applications)
    {
    }


    // Return the number of masses
    size_t size() const { return m_masses.size(); }

    // Return the sum total of all detected transforms. Equivalent to retrieving
    // `counts()` and summing, but more efficient.    
    size_t total_count() const { return m_total; }
    

    // Insert a new mass into the tracker and update counts.
    // Returns the id of the newly inserted mass (which by default is the same
    // as its insertion order).
    //
    // If `force_insert` is false, and `mass` is within tolerance of an existing
    // tracked mass, then `mass` is NOT inserted. Instead, the id of the nearest
    // matching mass is returned, and no counts are updated.
    // If `force_insert` is True, then `mass` is always inserted, and
    // counts are updated.
    //
    // Note: no check is done to make sure that user specified mass_ids are
    // unique. If they aren't, this could cause problems when `strict_count` is
    // true. Only use this if you have an external method to ensure uniqueness.
    
    size_t insert_mass(double mass, bool force_insert = false, int mass_id = -1) {
    auto existing = find_mass(mass);
    if (existing.has_value() && !force_insert) {
	return *existing;
    }

    size_t new_id;
    if (mass_id < 0) {
      new_id = m_masses.size();
    } else {
      if (m_mass_ids.contains(new_id)) {
        throw std::runtime_error("Mass id already exists");
      }
      new_id = mass_id;
    }      
    for (int xfrm_id = 0; xfrm_id < m_mass_deltas.size(); ++xfrm_id) {
      double delta = m_mass_deltas[xfrm_id];
      auto [lo_tol, hi_tol] = get_tols(mass, delta);

      // Check backward transform
      {
        auto [lo_it, hi_it] = find_close(mass - delta, lo_tol);
	update_counts(lo_it, hi_it, xfrm_id, false, new_id);	

      }        
      // Check forward transform
      {
        auto [lo_it, hi_it] = find_close(mass + delta, hi_tol);
	update_counts(lo_it, hi_it, xfrm_id, true, new_id);	
      }        
    }
    m_masses.insert({mass, new_id});
    m_mass_ids.insert({new_id, mass});
    m_min_mass = std::min(m_min_mass, mass);
    m_max_mass = std::max(m_max_mass, mass);
    return new_id;
  }

  template <typename T>
  void insert_masses(const T& masses, bool force_insert = false) {
    for (double m : masses) {
	insert_mass(m, force_insert);
    }      
  }
  
  // Check if `test_mass` is within tolerance of any tracked mass. If so,
  // return its id, otherwise return an empty optional.
  std::optional<int> find_mass(double test_mass) const {
      auto [lo_bound, hi_bound] = find_close(test_mass, test_mass * m_tol);
      if (lo_bound == hi_bound) {
	return std::nullopt;
      }
    return lo_bound->second;
  }

  // Returns the mass with the given id. Throws error if id does not correlate
  // to existing mass.
  double find_mass_by_id(size_t mass_id) const {
    auto it = m_mass_ids.find(mass_id);
    if (it == m_mass_ids.end()) {
	throw std::runtime_error("No mass with given id.");
    }
    return it->second;
  }

  
  bool contains_mass_id(size_t mass_id) const {
      return m_mass_ids.contains(mass_id);
  }    
  
  // For a given mass id and transform id, returns a pair of bools indicating
  // if the mass is the source (resp. dest) of that transform.
  // If mass_id is not valid, returns {false, false}  
  std::pair<bool, bool> contains_xfrm_by_id(size_t mass_id,
                                            size_t xfrm_id) const {
    return {
      m_seen.contains({mass_id, -xfrm_id - 1}),
      m_seen.contains({mass_id, xfrm_id})};          
  }


  std::pair<bool, bool> contains_xfrm_by_mass(double mass,
                                              size_t xfrm_id) const {
    auto mass_id = find_mass(mass);
    if (!mass_id.has_value()){
      return {false, false};
    }
    return contains_xfrm_by_id(*mass_id, xfrm_id);
  }    
  
  
  // Return an integer Eigen array of the current transformation counts. The
  // result has the same length as the `mass_deltas` used to initialize the
  // tracker, and the i'th entry of the result returns to the detected counts
  // for the i'th mass delta.  
  const IntArrayType &counts() const
      { return m_counts; }

  // Return all the applications detected so far. Each application is a triple
  // of the form (xfrm_id, src_id, dst_id), where 'xfrm_id` is the index of the
  // transformation in the `mass_deltas` used to initialize the tracker.
  // If the tracker was initialized with `track_applications=false`, this will
  // return an empty vector.
  // Note: the applications are not sorted in any way.  
    const applications_t& applications() const {
	return m_applications;
    }


    // Return all applications detected so far in a matrix form.
    // Applications are sorted first (by xrm_id, then src_id, then dst_id)    
    IntMatType sorted_applications() {
      std::sort(m_applications.begin(), m_applications.end());
      IntMatType result(m_applications.size(), 3);
      for (size_t ii = 0; ii < m_applications.size(); ++ii) {
	  result(ii, 0) = std::get<0>(m_applications[ii]);
	  result(ii, 1) = std::get<1>(m_applications[ii]);
	  result(ii, 2) = std::get<2>(m_applications[ii]);
      }
      return result;
    }




  // Return an array containing all masses in sorted order.  
  ArrayType masses() const {
    ArrayType result(m_masses.size());
    size_t idx = 0;
    for (const auto x : m_masses) {
      result(idx) = x.first;
      idx++;
    }      
    return result;
  }

  // Return an array of all mass ids, sorted by mass (that is, in the same
  // order as `masses()`.  
  IntArrayType mass_ids() const {
    IntArrayType result(m_masses.size());
    size_t idx = 0;
    for (const auto x : m_masses) {
      result(idx) = x.second;
      idx++;
    }      
    return result;
  }

  // Retrieve the list of mass deltas being tracked. This is identical to the
  // `mass_deltas` used to initialize the tracker.  
    ArrayType mass_deltas() const {
    ArrayType result(m_mass_deltas.size());
    size_t idx = 0;
    for (const auto x : m_mass_deltas) {
      result(idx) = x;
      idx++;
    }      
    return result;
    }

    // Retrieve the tolerance setting, in ppm    
    double tolerance_ppm() const { return m_tol * 1e6; }

    // Retrieve the `strict_count` setting.    
    bool strict_count() const { return m_strict; }

    // Retrieve the tolerance mode setting.    
    TransformMassMode mass_mode() const {return m_mode;}

    // Retrieve the track_applications setting.    
    bool track_applications() const {return m_track_applications;}
    

    // The following methods are exposed only to facilitate Python bindings,
    // and should not be used otherwise.    
  std::vector<std::pair<size_t, int>> get_seen() const {
    std::vector<std::pair<size_t, int>> result;
    result.reserve(m_seen.size());
    for (const auto &it : m_seen) {
      result.push_back(it);
    }
    return result;
  }

  std::map<double, size_t> get_massmap() const { return {m_masses.begin(), m_masses.end()}; }


  void set_state(std::map<double, size_t> masses,
                 std::vector<std::pair<size_t, int>> seen, IntArrayType counts,
                 const applications_t &applications
      ) {
    if (masses.size() > 0) {
        m_min_mass = std::min_element(masses.begin(), masses.end())->second;
        m_max_mass = std::max_element(masses.begin(), masses.end())->second;
      } else {        
	  m_min_mass = std::numeric_limits<double>::infinity();
          m_max_mass = -std::numeric_limits<double>::infinity();
      }
    m_masses.clear();
    for (const auto &it: masses) { 
      m_masses.insert(it);
    }
      m_mass_ids.clear();      
      for (const auto &it : m_masses) {
	  m_mass_ids.insert({it.second, it.first});
      }
      
      m_seen.clear();
      for (const auto &it : seen) {
        m_seen.insert(it);
      }
      m_total = 0;
      assert(m_counts.size() == counts.size());
      for (size_t ii = 0; ii < counts.size(); ++ii) {
        m_counts[ii] = counts[ii];
        m_total += counts[ii];
      }        
      m_applications = applications;
      
  }


private:

  std::pair <const_mass_iter_t,const_mass_iter_t>
      find_close(double test_mass, double tol_abs) const {
    // Find the range of masses that are within tol_abs of test_mass
      // If no masses found, return a pair of equal iterators
      if (test_mass < m_min_mass - tol_abs ||
	  test_mass > m_max_mass  + tol_abs)
	  return {m_masses.end(), m_masses.end()};
      auto it = m_masses.lower_bound(test_mass);
      auto hi_bound = it;
      auto lo_bound = it;
      // Now, if `it` is strictly within the set, then it lies between
      // two masses on either side of `test_mass`. If there are any close masses
      // then at least one of `it-1` or `it` points to one

      while (lo_bound != m_masses.begin() &&
             (abs(std::prev(lo_bound)->first - test_mass) < tol_abs)) {
	    lo_bound--;
      }

      // For hi_bound, we want to find the iterator one after the largest close
      // mass. If there are any
      
      while (hi_bound != m_masses.end() &&
             (abs(hi_bound->first - test_mass) < tol_abs)) {
	    hi_bound++;
        }

      return {lo_bound, hi_bound};
  }

  double get_average(double m1, double m2) const {
      return sqrt(m1 * m2);
  }

  std::pair<double, double> get_tols(double test_mass, double delta_m) {
    // Helper function to get the appropriate tolerances for testing
    // if a transformation is within the set.    
    switch (m_mode) {
    case (TransformMassMode::MODE_STRICT):
      return {m_tol * (test_mass - delta_m), m_tol * test_mass};
    case (TransformMassMode::MODE_LAX):
      return {m_tol * (test_mass), m_tol * (test_mass + delta_m)};
    case (TransformMassMode::MODE_MODERATE):
      return {
        m_tol *get_average(test_mass - delta_m, test_mass),
	m_tol * get_average(test_mass, test_mass + delta_m)};
    }
  }    

  void update_counts(const_mass_iter_t lo_it, const_mass_iter_t hi_it,
                     size_t xfrm_id, bool forward, size_t from_id) {
    if (lo_it == hi_it) {
      return;
    }

    int key = xfrm_id; // A quick way to encode both xfrm_id and direction
    if (!forward) {
	key = -key - 1;
    }
    
    // In the simple case, we just add a count for every transform we see:
    if (!m_strict) {
      while (lo_it != hi_it) {
	if (m_track_applications) {
	    if (forward) {
		m_applications.push_back({xfrm_id, from_id, lo_it->second});
	    } else {
		m_applications.push_back({xfrm_id, lo_it->second, from_id});
	    }
        }
        m_seen.insert({lo_it->second, key});
        m_seen.insert({from_id, -key - 1});
        
        m_counts[xfrm_id]++;
        m_total++;
        lo_it++;        
      }
    } else {
      // If we are being strict, we have to first make sure none of the
      // candidates have been seen before (combination of mass, transform,
	// and direction)
	
      bool found = false;
      auto it = lo_it;
      while (it != hi_it) {
	  if (m_seen.contains({it->second, key})) {
              found = true;
              break;
	  }
          it++;
      }          
      if (!found) {
	  m_seen.insert({lo_it->second, key});
	  if (m_track_applications) {
	      if (forward) {
                  m_applications.push_back({xfrm_id, from_id, lo_it->second});
	      } else {
		  m_applications.push_back({xfrm_id, lo_it->second, from_id});
	      }
	  }
	  m_seen.insert({lo_it->second, key});
	  m_seen.insert({from_id, -key - 1});
	  
	  m_counts[xfrm_id]++;
	  m_total++;
      }            
    }

  }
  
};


// Utility function for extracting transformations from a dataset.
IntMatType track_transforms(ConstArrayRef masses, ConstArrayRef mass_deltas, double err_ppm,
                            TransformMassMode err_mode);



} // namespace massim
#endif // MASSIM_MASS_TRACKER_H
