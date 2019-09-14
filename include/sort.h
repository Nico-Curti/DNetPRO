#ifndef __sort_h__
#define __sort_h__

#include <numeric>
#include <algorithm>

#ifdef _OPENMP

template < typename type, typename lambda >
void mergeargsort_serial (int * index, type * arr, const int & start, const int & end, lambda order);


template < typename type, typename lambda >
void mergeargsort_parallel_omp (int * index, type * arr, const int & start, const int & end, const int & threads, lambda order);

#endif

#endif // __sort_h__
