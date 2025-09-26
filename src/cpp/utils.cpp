#include "utils.h"

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

typedef std::tuple<double, int, int> diffrec_t;

std::vector<std::tuple<int, int, int>>
find_transforms(const VecType &masses, const VecType &xfrm_masses,
                double err_abs) {
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

  std::sort(std::execution::par_unseq, diffs.begin(), diffs.end());

  std::vector<std::tuple<int, int, int>> result;
  auto scan_it = diffs.begin();

  for (int xfrm_id = 0; xfrm_id < xfrm_masses.size(); ++xfrm_id) {
    // Find the range of differences that fall within `err_abs` of their
    // transform mass.
    double delta = xfrm_masses[xfrm_id];
    scan_it = std::lower_bound(scan_it, diffs.end(),
                               diffrec_t{delta - err_abs, 0, 0});

    // Once we've found the start, scan forward until the transform no longer
    // matches

    while (scan_it != diffs.end() && std::get<0>(*scan_it) < delta + err_abs) {
      result.push_back(std::make_tuple(xfrm_id, std::get<1>(*scan_it),
                                       std::get<2>(*scan_it)));
      ++scan_it;
    }
  }

  return result;
}

} // namespace massim
