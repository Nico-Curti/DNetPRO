#ifndef __sort_hpp__
#define __sort_hpp__

#include <sort.h>


#ifdef _OPENMP

#define __minimum_sort_size__ 1000

template < typename type, typename lambda >
void mergeargsort_serial (int32_t * index, type * /*arr*/, const int32_t & start, const int32_t & end, lambda order)
{
  if ((end - start) == 2)
  {
    if (order(index[start] , index[end - 1]))
      return;
    else
    {
      std :: swap(index[start], index[end - 1]);
      return;
    }
  }

  const int32_t pivot = start + ((end - start) >> 1);

  if ((end - start) < __minimum_sort_size__)
  {
    std :: sort(index + start, index + end, order);
    return;
  }
  else
  {
    mergeargsort_serial(index, /*arr*/ nullptr, start, pivot, order);
    mergeargsort_serial(index, /*arr*/ nullptr, pivot, end, order);
  }

  std :: inplace_merge(index + start, index + pivot, index + end, order);

  return;
}

template < typename type, typename lambda >
void mergeargsort_parallel_omp ( int32_t * index, type * arr, const int32_t & start, const int32_t & end, const int32_t & threads, lambda order)
{
  const int32_t pivot = start + ((end - start) >> 1);

  if (threads <= 1 || (end - start) < __minimum_sort_size__)
  {
    mergeargsort_serial(index, /*arr*/ nullptr, start, end, order);
    return;
  }
  else
  {
#pragma omp task firstprivate (start, pivot, threads) shared (index, order)
    {
      mergeargsort_parallel_omp(index, /*arr*/ nullptr, start, pivot, threads >> 1, order);
    }
#pragma omp task firstprivate (pivot, end, threads) shared (index, order)
    {
      mergeargsort_parallel_omp(index, /*arr*/ nullptr, pivot, end, threads - (threads >> 1), order);
    }
#pragma omp taskwait
  }

  std :: inplace_merge(index + start, index + pivot, index + end, order);

  return;
}


#endif // omp

#endif // __sort_hpp__
