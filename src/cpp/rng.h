#ifndef MASSIM_RNG_H
#define MASSIM_RNG_H

#include <random>
#include "common.h"

namespace massim {

const int RANDOM_SEED = std::numeric_limits<int>::max();

using RandomEngine_t = std::mt19937;

struct RNG {
  RandomEngine_t engine;

  RNG() {
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    engine = RandomEngine_t(seed);
  }

  RNG(int seed) : engine(seed) {}

  // Spawns a new RNG from an existing one
  RNG(RNG &rng) {
    std::seed_seq seed{rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng()};
    engine = RandomEngine_t(seed);
  }

  static RNG &default_rng() {
    static RNG rng;
    return rng;
  }

  RNG(RNG *rng, bool spawn = true) {
    if (rng == nullptr) {
      rng = &RNG::default_rng();
    }
    if (spawn) {
      RNG &r = *rng;
      std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
      engine = RandomEngine_t(seed);
    } else {
      engine = rng->engine;
    }
  }

  RandomEngine_t::result_type operator()() { return engine(); }
};

}  // namespace massim

#endif  // MASSIM_RNG_H
