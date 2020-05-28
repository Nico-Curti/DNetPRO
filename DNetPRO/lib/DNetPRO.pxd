# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr
from libcpp cimport bool

cdef extern from "score.h":

  cppclass score:

    score () except +
    score (const int & N, const int & n_class) except +

    score (score & s) except +

    ## Attributes

    unique_ptr[float[]] mcc;
    unique_ptr[int[]] gene_a;
    unique_ptr[int[]] gene_b;
    unique_ptr[int[]] tot;

    int N;
    int n_class;

cdef extern from "dnetpro_couples.h":

  score dnetpro_couples (float ** data,
                         const int & Nprobe,
                         const int & Nsample,
                         int * labels,
                         const bool & verbose,
                         float percentage,
                         # const bool & return_couples,
                         int nth)

cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[score] move(unique_ptr[score])


cdef class _score:

  cdef unique_ptr[score] thisptr

  cdef public:
    int N;
    int n_class;

  cdef inline _move (self, score src):
    self.N, self.n_class = src.N, src.n_class
    self.thisptr.reset(new score(src))
