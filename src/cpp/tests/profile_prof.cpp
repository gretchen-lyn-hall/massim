#include <ostream>
#include <vector>
#include <iostream>

#include "../common.h"
#include "../mass_transformer.h"
#include "../profile_generator.h"
#include "../distribution.h"

using massim::ArrayType;

std::vector<double> mass_deltas = {
    0.984016,   1.003355,   1.995664,   2.01565,    2.03946,    14.003074,
    14.01565,   15.010899,  15.977157,  17.966113,  27.994915,  30.973762,
    44.013639,  61.915598,  67.054775,  79.956815,  79.966331,  87.032028,
    110.035437, 132.042259, 134.04667,  137.058912, 147.068414, 156.101111,
    160.940487, 162.052823, 177.045964, 186.079313, 224.079707, 226.058971,
    226.077599, 227.157644, 229.014009, 238.229666, 243.080339, 249.086189,
    306.075981, 427.0199,   448.061023, 486.15847,  748.096821, 765.09956};

std::vector<double> chain_prob = {
    0.01423,  0.01441,  0.010508, 0.1642,   0.016599, 0.019,    0.122323,
    0.012504, 0.017011, 0.016847, 0.140338, 0.004457, 0.016789, 0.006105,
    0.006735, 0.014517, 0.013822, 0.006655, 0.00562,  0.182355, 0.00436,
    0.002423, 0.002748, 0.005498, 0.000374, 0.130491, 0.006049, 0.003972,
    0.001534, 0.001348, 0.001703, 0.000612, 0.000241, 0.118629, 0.000412,
    0.000806, 0.00169,  0.000106, 7.9e-05,  0.000683, 8e-06,    8e-06};

std::vector<double> chain_len = {
    2.034796, 2.068367, 2.116273, 4.15917,  2.019691, 2.021418, 5.641581,
    2.02673,  2.035842, 2.013511, 4.337374, 2.042421, 2.011219, 2.015019,
    2.149489, 2.01186,  2.015336, 2.014828, 2.012195, 2.095078, 2.016646,
    2.056466, 2.04177,  2.00804,  2.026204, 2.059958, 2.002962, 2.014307,
    2.017979, 2.012821, 2.023546, 2.013832, 2.007752, 2.065359, 2.009714,
    2.023201, 2.078796, 2.015625, 2.040816, 2.304131, 2.0,      2.0};



int main(int argc, char** argv) {
  massim::TransformList tf_list(Eigen::Map<ArrayType>(mass_deltas.data(), mass_deltas.size()),
				Eigen::Map<ArrayType>(chain_prob.data(), chain_prob.size()),
				Eigen::Map<ArrayType>(chain_len.data(), chain_len.size()),
				Eigen::Map<ArrayType>(chain_len.data(), chain_len.size()),
				false, false);

  massim::UniformDistribution unif(0,1.0);
  massim::RNG rng;
  
  for (int ii=0; ii<40; ++ii) {
    massim::ProfileGenerator pf(500, tf_list, unif);
    for (int pid=0; pid < pf.num_profiles(); ++pid) {
      pf.update_component(pid, 400.0, 1.0);
    }

    pf.apply_transforms(20000, rng);
    std::cout << "Finished profile " << ii << std::endl << std::flush;
    
  }

}
