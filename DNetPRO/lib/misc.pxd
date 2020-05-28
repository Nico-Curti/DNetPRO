# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr

cdef extern from "misc.hpp":

  type * unique_pointer_to_pointer[type] (const unique_ptr[type[]] & src, const int & size);
  unique_ptr[type[]] pointer_to_unique_pointer[type] (type * src, const int & size);
  type_out ** two_dimension_pointer_for_cython[type_in, type_out](type_in * input, const int & n_row, const int & n_cols);

