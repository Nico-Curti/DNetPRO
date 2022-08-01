#ifndef __sort_hpp__
#define __sort_hpp__

#include <sort.h>


#ifdef _OPENMP

#define __minimum_sort_size__ 1000

template < typename type, typename lambda >
void mergeargsort_serial (int32_t * index, type * arr, const int32_t & start, const int32_t & end, lambda order)
{
  if ((end - start) == 2)
  {
    if (order(arr[start] , arr[end - 1]))
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
    mergeargsort_serial(index, arr, start, pivot, order);
    mergeargsort_serial(index, arr, pivot, end, order);
  }

  std :: inplace_merge(index + start, index + pivot, index + end, order);

  return;
}

template < typename type, typename lambda >
void mergeargsort_parallel_omp ( int32_t * index, type * arr, const int32_t & start, const int32_t & end, const int32_t & threads, lambda order)
{
  const int32_t pivot = start + ((end - start) >> 1);

  if (threads <= 1)
  {
    mergeargsort_serial(index, arr, start, end, order);
    return;
  }
  else
  {
#pragma omp task shared (start, end, threads)
    {
      mergeargsort_parallel_omp(index, arr, start, pivot, threads >> 1, order);
    }
#pragma omp task shared (start, end, threads)
    {
      mergeargsort_parallel_omp(index, arr, pivot, end, threads - (threads >> 1), order);
    }
#pragma omp taskwait
  }

  std :: inplace_merge(index + start, index + pivot, index + end, order);

  return;
}


#endif // omp

#endif // __sort_hpp__
