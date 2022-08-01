# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr
cimport numpy as np

cdef extern from "misc.hpp":

  type * unique_pointer_to_pointer[type] (const unique_ptr[type[]] & src, const np.int32_t & size);
  unique_ptr[type[]] pointer_to_unique_pointer[type] (type * src, const np.int32_t & size);
  type_out ** two_dimension_pointer_for_cython[type_in, type_out](type_in * input, const np.int32_t & n_row, const np.int32_t & n_cols);

