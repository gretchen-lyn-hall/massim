#include "mass_tracker.h"
#include <cassert>

namespace massim {

IntMatType track_transforms(ConstArrayRef masses, ConstArrayRef mass_deltas,
                            double err_ppm, TransformMassMode err_mode) {
  MassTracker tracker(mass_deltas, err_ppm, err_mode, /*strict=*/false,
                      /*track_applications=*/true);
  for (auto mass : masses) {
    tracker.insert_mass(mass, /*force_insert=*/true);
  }
  return tracker.sorted_applications();
}

} // namespace massim
