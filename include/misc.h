#ifndef __dnetpro_misc__
#define __dnetpro_misc__

#ifdef __dnet__

#include <memory>    // std :: unique_ptr
#include <algorithm> // std :: copy_n
#include <utility>   // std :: move

/**
* @brief Cython utility to convert unique_pointer to simple pointer
*
* @tparam type type of the pointer array
*
* @param src input smart pointer to convert
* @param size lenght of unique_ptr array
*
* @returns pointer array as simple pointer
*
*/
template < typename type >
type * unique_pointer_to_pointer (const std :: unique_ptr < type[] > & src, const std :: size_t & size);

/**
* @brief Cython utility to convert simple pointer to unique_pointer
*
* @tparam type type of the pointer array
*
* @param src input pointer to convert
* @param size lenght of unique_ptr array
*
* @returns pointer array as unique_pointer
*
*/
template < typename type >
std :: unique_ptr < type[] > pointer_to_unique_pointer (type * src, const std :: size_t & size);

/**
* @brief Cython utility to convert ravel simple pointer to 2D pointer matrix
*
* @tparam type_in type of the input pointer array
* @tparam type_out type of the output pointer 2D
*
* @param input input pointer to convert
* @param n_row number of rows of the output matrix
* @param n_cols number of columns of the output matrix
*
* @returns pointer array as 2D pointer
*
*/
template < typename type_in, typename type_out >
type_out ** two_dimension_pointer_for_cython (type_in * input, const int32_t & n_row, const int32_t & n_cols);

#endif // dnet

#endif // __dnetpro_misc__
