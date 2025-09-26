#ifndef MASSIM_UTILS_H
#define MASSIM_UTILS_H

#include "common.h"

namespace massim
{

  int adjust_masses(
		    Eigen::Ref<ArrayType> masses,
			const ArrayType& valid_masses);



// Given a (sorted) list of compound masses and a list of transformation mass
// deltas, find all potential transformations.
// The output is a vector of <transform_idxs, src_idx, dst_idx>  
std::vector<std::tuple<int, int, int>> find_transforms(const VecType& masses,
	const VecType& xfrm_masses,
	double err_abs);



}  // namespace massim
#endif  // MASSIM_UTILS_H


