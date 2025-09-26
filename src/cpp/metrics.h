#ifndef MASSIM_METRICS
#define MASSIM_METRICS

#include "common.h"

namespace massim
{

  MatType compute_aff(const Eigen::Ref<const MatType>& sim);

  MatType jaccard(const Eigen::Ref<const BoolMatType>& presence);

  MatType jaccard(const Eigen::Ref<const MatType>& presence);

  MatType braycurtis(const Eigen::Ref<const MatType>& mat);



}  // namespace massim
#endif  // MASSIM_METRICS
