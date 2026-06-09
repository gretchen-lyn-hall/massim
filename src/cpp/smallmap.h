#ifndef SMALLMAP_H
#define SMALLMAP_H

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>

const size_t tab64[64] = {
    63,  0, 58,  1, 59, 47, 53,  2,
    60, 39, 48, 27, 54, 33, 42,  3,
    61, 51, 37, 40, 49, 18, 28, 20,
    55, 30, 34, 11, 43, 14, 22,  4,
    62, 57, 46, 52, 38, 26, 32, 41,
    50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16,  9, 12,
    44, 24, 15,  8, 23,  7,  6,  5};

size_t log2_64 (uint64_t value)
{
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
}

template <typename Key, typename Value>
class smallmap {
  typedef std::pair<Key, Value> item_t;
  typedef std::vector<std::pair<Key, Value>> sm_t;

  sm_t m_vec;
  int m_sorted_end;
  size_t m_next_sort;
  const item_t m_end_item;

  struct KeyCompare {
    bool operator()(const item_t& item, Key value) {
      return item.first < value;
    }
  };

public:

  struct iterator {
    const smallmap& m_map;
    const item_t& m_item;
    // Keep track of both the current location, and
    // the lower bound of this value in the sorted region (same if in sorted)
    // That is, the largest sorted item less than or equal to this
    typename sm_t::iterator m_loc;
    typename sm_t::iterator m_sorted_lb;

    item_t& operator*() {
      return m_item;
    }

    bool operator==(const smallmap<Key, Value>::iterator& rhs) {
      return (&m_map == &rhs.m_map && m_loc == rhs.m_loc);
    }

  };

  // void find_next(typename sm_t::iterator& it,
  // 		 typename sm_t::iterator& it_lb) {
  //   if (it == m_vec.end())
  //     return;
  //   if (m_sorted_end == m_vec.end()) {
  //     it++;
  //     it_lb++;
  //   } else {
  //     // Find the next largest item in the unsorted region
  //     typename sm_t::iterator& best_unsorted = m_sorted_end;
  //     for (const typename sm_t::iterator un_it = m_sorted_end;
  // 	   un_it < m_vec.end(); ++un_it) {
  // 	if (un_it->first > it->first) {
  // 	  // This first case only can happen once
  // 	  if (best_unsorted->first <= it->first) {
  // 	    best_unsorted = un_it;
  // 	  } else if ( best_unsorted->first > un_it->first) {
  // 	    best_unsorted = un_it;
  // 	  }
  // 	}
  //     }
  //     if (best_unsorted->first <= it.first)
      
      
  // }
   
  
  smallmap(size_t reserve=0):
    m_next_sort(32),
    m_end_item({std::numeric_limits<Key>::max(), Value()})  {
    m_vec.reserve(reserve);
    m_sorted_end = 0;
  }

  int bisect_sorted(double key) const {
     int lo = 0;
     int hi = m_sorted_end - 1;
     if (hi < 0 || key <= m_vec[lo].first)
       return lo;
     auto p = &m_vec[0];
    if (key >= m_vec[hi].first) {
      if (key == m_vec[hi].first)
	return hi;
      return hi + 1;
    }
      
    while (lo < hi - 1) {
      int mid = (lo + hi)  >> 1;
      if ((p+mid)->first < key) {
	lo = mid;
      } else {
	hi = mid;
      }      
    }
    return hi;
  }

  void insert(Key key, Value value) {
    // Check for existence
    typename sm_t::iterator sorted_end = m_vec.begin() + m_sorted_end;
    typename sm_t::iterator sorted_it = m_vec.begin() + bisect_sorted(key);

    if (sorted_it != sorted_end && sorted_it->first == key) {
      sorted_it->second = value;
      return;
    }
    for (typename sm_t::iterator it = sorted_end; it < m_vec.end(); ++it) {
      if (it->first == key) {
	it->second = value;
	return;
      }
    }

    insert_nocheck(key, value);
  }

  void insert_nocheck(Key key, Value value) {
    m_vec.emplace_back(std::make_pair(key, value));
    if (m_vec.size() > m_next_sort) {
      resort();
    } else if (m_vec.size() == 1) {
      // Special case - make sure the sorted region is never empty if the
      // vector isn't. An array of size 1 is *always* sorted!
      m_sorted_end = 1;
    }
  }


  smallmap::iterator begin() {
    if (m_vec.size() >0)
      return iterator(m_vec, m_vec.begin());
    return end();
  }

  smallmap::iterator end() {
    return iterator(m_vec, m_vec.end(), m_end_item);
  }

  const item_t& lower_bound(const Key key) const {
    if (m_vec.size() == 0) {
      return m_end_item;
    }
    
    typename sm_t::const_iterator sorted_end = m_vec.begin() + m_sorted_end;
    typename sm_t::const_iterator cur_best = m_vec.end();

    for (typename sm_t::const_iterator unsorted_it = sorted_end;
	 unsorted_it < m_vec.end(); ++unsorted_it) {
 
      // If in loop, cur_best points to a valid item
      if (unsorted_it->first >= key) {
	if (cur_best == m_vec.end()) {
	  cur_best = unsorted_it;
	} else if (unsorted_it->first < cur_best->first)
	  cur_best = unsorted_it;
      }
    }
    
    typename sm_t::const_iterator sorted_it =  m_vec.begin() + bisect_sorted(key);
    
    // Lotta cases:
        
    if (sorted_it != sorted_end) {
      if (cur_best == m_vec.end()) {
	return *sorted_it;
      }
      if (sorted_it->first < cur_best->first) {
	return *sorted_it;
      }
      return *cur_best;
    } else {
      if (cur_best == m_vec.end()) {
	return m_end_item;
      }

      return *cur_best;
    }
  }

private:
  void resort() {
    typename sm_t::iterator sorted_end = m_vec.begin() + m_sorted_end;

    std::sort(sorted_end, m_vec.end());
    std::inplace_merge(m_vec.begin(), sorted_end, m_vec.end());
    m_sorted_end = m_vec.size();
    m_next_sort = m_vec.size()  + log2_64(m_vec.size());
  }
};

  


#endif
