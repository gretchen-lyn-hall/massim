#include "utils.h"
#include <iostream>
#include <queue>
#include <execution>
namespace massim {

int adjust_masses(Eigen::Ref<ArrayType> masses, const ArrayType &valid_masses) {
  int badcount = 0;
  auto v_begin = valid_masses.begin();
  auto v_end = valid_masses.end();
  for (int idx = 0; idx < masses.size(); ++idx) {
    auto it = std::lower_bound(v_begin, v_end, masses(idx));
    if (it == v_end) {
      badcount++;
    } else {
      masses(idx) = *it;
    }
  }
  return badcount;
}


using  heapent_t = std::pair<double, size_t>;

    
difflist_t diffsort(const ArrayType &masses) {
  int N2 = masses.size() * (masses.size() - 1) / 2;
  std::vector<ArrayType> rows;
  std::priority_queue<heapent_t, std::vector<heapent_t>, decltype(std::greater<heapent_t>())> heap;  
  difflist_t result(N2);
  VecType last_row = masses;
  VecType next_row;
  rows.reserve(masses.size() - 1);
      
  if (!std::is_sorted(masses.begin(), masses.end())) {
    throw std::invalid_argument("Input masses must be pre-sorted");
  }


  // Prepare the rows. We have to be careful not to overwrite a row while
  // performing calculations.  
  while (last_row.size() > 1) {
    
      next_row = last_row.tail(last_row.size()-1).array() - last_row(0);
      rows.push_back(next_row);
      last_row = next_row;
  }
  // Position along each row
  std::vector<size_t> indices(rows.size(), 0);

  // Initialize the heap
  for (int ii=0; ii<rows.size(); ++ii) {
      heap.push({rows[ii](0), ii});
  }

  size_t ctr = 0;

  while (!heap.empty()) {
    auto [diff, row_id] = heap.top();
    heap.pop();
    auto &idx = indices[row_id];
    result[ctr] = {diff, row_id, idx};
    if (++idx < rows[row_id].size()) {
	heap.push({rows[row_id][idx], row_id});
    }
    ctr += 1;
  }    
  
  return result;
}

difflist_t simplediffsort(const ArrayType &masses) {
    int N2 = masses.size() * (masses.size() - 1) / 2;
    std::vector<diffrec_t> diffs;
    diffs.reserve(N2);

    for (int ii = 0; ii < masses.size() - 1; ++ii) {
	for (int jj = ii + 1; jj < masses.size(); ++jj) {
	    double diff = masses.coeff(jj) - masses.coeff(ii);
		diffs.push_back({diff, ii, jj});
        }
    }
    #ifdef __APPLE__
    std::sort(diffs.begin(), diffs.end());
    #else
    std::sort(std::execution::par_unseq, diffs.begin(), diffs.end());
    #endif
    return diffs;
}    


std::vector<std::tuple<int, int, int>>
find_transforms(const VecType &masses, const VecType &xfrm_masses,
                double err_abs, bool allow_overlap) {
  if (!std::is_sorted(masses.begin(), masses.end())) {
    throw std::invalid_argument("Input masses must be pre-sorted");
  }

  if (!std::is_sorted(xfrm_masses.begin(), xfrm_masses.end())) {
    throw std::invalid_argument("Input transforms must be pre-sorted");
  }

  int N2 = masses.size() * (masses.size() - 1) / 2;
  std::vector<diffrec_t> diffs;
  diffs.reserve(N2);
  double min_delta = xfrm_masses.minCoeff() - err_abs;
  double max_delta = xfrm_masses.maxCoeff() + err_abs;

  for (int ii = 0; ii < masses.size() - 1; ++ii)
    for (int jj = ii + 1; jj < masses.size(); ++jj) {
      double diff = masses.coeff(jj) - masses.coeff(ii);
      if (diff > min_delta && diff < max_delta)
        diffs.push_back({diff, ii, jj});
    }

#ifdef __APPLE__
  std::sort(diffs.begin(), diffs.end());
#else
  std::sort(std::execution::par_unseq, diffs.begin(), diffs.end());
#endif
  
  std::vector<std::tuple<int, int, int>> result;
  auto scan_start = diffs.begin();

  for (int xfrm_id = 0; xfrm_id < xfrm_masses.size(); ++xfrm_id) {
    // Find the range of differences that fall within `err_abs` of their
    // transform mass.
    double delta = xfrm_masses[xfrm_id];
    auto scan_it = std::lower_bound(scan_start, diffs.end(),
				    diffrec_t{delta - err_abs, 0, 0});
    scan_start = scan_it;
    
    // Once we've found the start, scan forward until the transform no longer
    // matches

    while (scan_it != diffs.end() && std::get<0>(*scan_it) < delta + err_abs) {
      result.push_back(std::make_tuple(xfrm_id, std::get<1>(*scan_it),
                                       std::get<2>(*scan_it)));
      ++scan_it;
    }
    if (!allow_overlap) {
	// Start the next scan at the end of this one
	scan_start = scan_it;
    }          
    
  }

  return result;
}

} // namespace massim
