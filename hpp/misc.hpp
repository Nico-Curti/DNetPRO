#ifndef __dnetpro_misc_hpp__
#define __dnetpro_misc_hpp__

#include <misc.h>

#ifdef __dnet__

template < typename type >
type * unique_pointer_to_pointer (const std :: unique_ptr < type[] > & src, const std :: size_t & size)
{
  type * dest = new type[size];
  std :: copy_n(src.get(), size, dest);
  return dest;
}

template < typename type >
std :: unique_ptr < type[] > pointer_to_unique_pointer (type * src, const std :: size_t & size)
{
  std :: unique_ptr < type[] > dest (new type[size]);
  std :: move(src, src + size, dest.get());
  return dest;
}


template < typename type_in, typename type_out >
type_out ** two_dimension_pointer_for_cython (type_in * input, const int & n_row, const int & n_cols)
{
  type_out ** output = new type_out * [n_row];
  std :: generate_n (output, n_row, [&] () { return new type_out[n_cols]; });

  int idx = 0;
  for (int i = 0; i < n_row; ++i)
    for (int j = 0; j < n_cols; ++j, ++idx)
      output[i][j] = static_cast < type_in >(input[idx]);

  return output;
}


#endif

#endif // __dnetpro_misc_hpp__
