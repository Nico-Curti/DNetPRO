# distutils: language = c++
# cython: language_level=3

from libcpp.memory cimport unique_ptr
cimport numpy as np
from libc.stdint cimport int32_t
from libcpp cimport bool as cpp_bool

cdef extern from "score.h":

  cppclass score:

    score () except +
    score (const np.int32_t & N, const np.int32_t & n_class) except +

    score (score & s) except +

    ## Attributes

    unique_ptr[float[]] mcc;
    unique_ptr[np.int32_t[]] gene_a;
    unique_ptr[np.int32_t[]] gene_b;
    unique_ptr[np.int32_t[]] tot;
    unique_ptr[np.int32_t[]] class_score;

    np.int32_t N;
    np.int32_t n_class;

cdef extern from "dnetpro_couples.h":

  unique_ptr[score] dnetpro_couples (
    float ** data,
    const int32_t & Nprobe,
    const int32_t & Nsample,
    int32_t * labels,
    const cpp_bool & verbose,
    float percentage,
    int32_t nth)

cdef extern from "<utility>" namespace "std" nogil:
  cdef unique_ptr[score] move(unique_ptr[score])

cdef class _score:

  cdef unique_ptr[score] thisptr

  cdef public:
    np.int32_t N;
    np.int32_t n_class;

    cdef void _set_ptr(self, unique_ptr[score] & ptr);
