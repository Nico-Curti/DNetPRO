#include <misc.hpp>

#ifdef __dnet__

template int * unique_pointer_to_pointer < int > (const std :: unique_ptr < int[] > & src, const std :: size_t & size);

template float * unique_pointer_to_pointer < float > (const std :: unique_ptr < float[] > & src, const std :: size_t & size);

template float ** two_dimension_pointer_for_cython < float, float > (float * input, const int & n_row, const int & n_cols);

#endif
