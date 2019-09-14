#ifndef __dnetpro_misc__
#define __dnetpro_misc__

#ifdef __dnet__

#include <memory>    // std :: unique_ptr
#include <algorithm> // std :: copy_n
#include <utility>   // std :: move

template < typename type >
type * unique_pointer_to_pointer (const std :: unique_ptr < type[] > & src, const std :: size_t & size);

template < typename type >
std :: unique_ptr < type[] > pointer_to_unique_pointer (type * src, const std :: size_t & size);

template < typename type_in, typename type_out >
type_out ** two_dimension_pointer_for_cython (type_in * input, const int & n_row, const int & n_cols);

#endif // dnet

#endif // __dnetpro_misc__
