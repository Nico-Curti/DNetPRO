#ifndef __sort_h__
#define __sort_h__

#include <numeric>
#include <algorithm>

#ifdef _OPENMP

/**
* @brief Mergesort of the indexes serial
*
* @details This function is the serial version of the sorting using indexes.
* The original array is unchanged at the end of the function but the indexes for its reordering
* are evaluated and stored into the `indexes` variable.
* See numpy.argsort in Python as analogy.
*
* @tparam type Input array type
* @tparam lambda Function for the comparison evaluation
*
* @param index Array of indexes (it must have the same size of `arr`)
* @param arr Array of values to reorder
* @param start Starting point for the reordering (commonly set to `0`)
* @param end End point for the reordering (commonly set to `array_size`)
* @param order Lambda function for the comparison evaluation
*
*
*/
template < typename type, typename lambda >
void mergeargsort_serial (int32_t * index, type * arr, const int32_t & start, const int32_t & end, lambda order);

/**
* @brief Mergesort of the indexes with OMP multithreading support
*
* @details This function is the parallel version of the sorting using indexes.
* The original array is unchanged at the end of the function but the indexes for its reordering
* are evaluated and stored into the `indexes` variable.
* See numpy.argsort in Python as analogy.
*
* @tparam type Input array type
* @tparam lambda Function for the comparison evaluation
*
* @param index Array of indexes (it must have the same size of `arr`)
* @param arr Array of values to reorder
* @param start Starting point for the reordering (commonly set to `0`, but it is used for the parallelization master-slave)
* @param end End point for the reordering (commonly set to `array_size`)
* @param threads Number of threads to use in the evaluation
* @param order Lambda function for the comparison evaluation
*
*
*/
template < typename type, typename lambda >
void mergeargsort_parallel_omp (int32_t * index, type * arr, const int32_t & start, const int32_t & end, const int32_t & threads, lambda order);

#endif

#endif // __sort_h__
