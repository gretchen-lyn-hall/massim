#include <random>
#include <ctime>

#include <map>
#include <unordered_map>
#include <iostream>
#include "../flat_map.h"
#include <format>

#include "../smallmap.h"
#include "../rng.h"

#include <chrono>

using std::chrono::steady_clock;
using std::chrono::duration_cast;

typedef std::chrono::time_point<steady_clock> time_point;

double timediff(time_point then, time_point now) {
  return duration_cast<std::chrono::nanoseconds>(now - then).count() / 1000000.0;
}



const size_t N_REPS = 100085;


int main(int argc, char** argv) {
  std::uniform_int_distribution<> valgen(0, 1000);
  std::uniform_real_distribution<> keygen(0, 1);
  massim::RNG rng;
  
    
  std::map<double, int> st_map;
  stdext::flat_map<double, int> st_hash;
  smallmap<double, int> my_map(N_REPS);


  time_point start = steady_clock::now();
  
  for (int ii=0; ii<N_REPS; ++ii) {
    double key = keygen(rng.engine);
    if (key==7) {
      std::cout << "Gotcha1" << std::endl;
    }
  }
  std::cout << "Just random: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;



  start = steady_clock::now();
  for (int ii=0; ii<N_REPS; ++ii) {
    st_map.insert({keygen(rng.engine), valgen(rng.engine)});
  }
  
  std::cout << "Standard map: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;


  start = steady_clock::now();
  for (int ii=0; ii<N_REPS; ++ii) {
    st_hash.insert({keygen(rng.engine), valgen(rng.engine)});
  }
  
  std::cout << "Standard hash: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;

  start = steady_clock::now();
  for (int ii=0; ii<N_REPS; ++ii) {
    my_map.insert_nocheck(keygen(rng.engine), valgen(rng.engine));
  }

  std::cout << "SmallMap: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;
  
  std::cout << "\nLookup:" << std::endl;
  
  start = steady_clock::now();
  for (int ii=0; ii<N_REPS; ++ii) {
    std::map<double, int>:: iterator it = st_map.lower_bound(keygen(rng.engine));
    if (it->second == 100000) {
      // Force optimizer to actually do the lookup
      std::cout << "Gotcha!";
    }
  }
  std::cout << "Standard map: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;

  start = steady_clock::now();
  for (int ii=0; ii<N_REPS; ++ii) {
    auto it = st_hash.lower_bound(keygen(rng.engine));
    const std::pair<double, int>& result = *it;
  }
  std::cout << "Standard hash: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;


  
  start = steady_clock::now();
  for (int ii=0; ii<N_REPS; ++ii) {
    auto val = my_map.bisect_sorted(keygen(rng.engine));
    if (val == 10000000)
      std::cout << "Gotcha!" << std::endl;
  }
  std::cout << "SmallMap: "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;

  start = steady_clock::now();

  std::cout << "Correctness" << std::endl;
  std::map<double, int> st_map2;
  smallmap<double, int> my_map2;
  for (int ii=0; ii<N_REPS; ++ii) {
    double key = keygen(rng.engine);
    int val = valgen(rng.engine);
    st_map2.insert({key, val});
    my_map2.insert(key, val);
  }

  time_point end;
  for (int ii=0; ii<N_REPS; ++ii) {
    double key = keygen(rng.engine);
    int val = valgen(rng.engine);
    std::map<double, int>:: iterator it = st_map2.lower_bound(key);
    const std::pair<double, int>& result = my_map2.lower_bound(key);
    if (it != st_map2.end()) {
      if (it->first != result.first) {
	end = steady_clock::now();
	std::cout << "UhOh! On test iteration "<< ii << "with key" << key <<std::endl;
	std::cout << "Standard map found" <<  it->first << std::endl;
	std::cout << "Smallmap found" <<  result.first << std::endl;
	break;
      }
    }
    
  }
  std::cout << "Both "
	    << timediff(start, steady_clock::now())
	    << "ms" << std::endl;


  

}
