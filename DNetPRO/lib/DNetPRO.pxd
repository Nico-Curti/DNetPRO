# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr
from libcpp cimport bool
cimport numpy as np

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

    np.int32_t N;
    np.int32_t n_class;

cdef extern from "dnetpro_couples.h":

  score dnetpro_couples (float ** data,
                         const np.int32_t & Nprobe,
                         const np.int32_t & Nsample,
                         np.int32_t * labels,
                         const bool & verbose,
                         float percentage,
                         # const bool & return_couples,
                         np.int32_t nth)

cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[score] move(unique_ptr[score])


cdef class _score:

  cdef unique_ptr[score] thisptr

  cdef public:
    np.int32_t N;
    np.int32_t n_class;

  cdef inline _move (self, score src):
    self.N, self.n_class = src.N, src.n_class
    self.thisptr.reset(new score(src))
