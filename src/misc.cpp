#include <misc.hpp>

#ifdef __dnet__

template int32_t * unique_pointer_to_pointer < int32_t > (const std :: unique_ptr < int32_t[] > & src, const std :: size_t & size);

template float * unique_pointer_to_pointer < float > (const std :: unique_ptr < float[] > & src, const std :: size_t & size);

template float ** two_dimension_pointer_for_cython < float, float > (float * input, const int32_t & n_row, const int32_t & n_cols);

#endif
