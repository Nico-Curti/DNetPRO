# distutils: language = c++
# cython: language_level=2

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
cimport numpy as np

from DNet_couples cimport score
from DNet_couples cimport dnetpro_couples
from DNet_couples cimport move

from misc cimport unique_pointer_to_pointer
from misc cimport pointer_to_unique_pointer
from misc cimport two_dimension_pointer_for_cython

import numpy as np

cdef class _score:

  cdef unique_ptr[score] thisptr

  cdef public:
    int N;
    int n_class;

  def __init__ (self, other=None):

    try:

      self.N, self.n_class = other
      self.thisptr.reset(new score(self.N, self.n_class))

    except TypeError:

      self.N, self.n_class = (0, 0)
      self.thisptr.reset(new score())

  cdef _move (self, score src):
    self.N, self.n_class = src.N, src.n_class
    self.thisptr.reset(new score(src))

  def __str__ (self):
    return '<Score: N={} n_class={}>'.format(deref(self.thisptr).N, deref(self.thisptr).n_class)


  @property
  def mcc (self):
    cdef float * mcc = unique_pointer_to_pointer(deref(self.thisptr).mcc, self.N)
    return [mcc[i] for i in range(self.N)]

  @property
  def gene_a(self):
    cdef int * gene_a = unique_pointer_to_pointer(deref(self.thisptr).gene_a, self.N)
    return [gene_a[i] for i in range(self.N)]

  @property
  def gene_b(self):
    cdef int * gene_b = unique_pointer_to_pointer(deref(self.thisptr).gene_b, self.N)
    return [gene_b[i] for i in range(self.N)]

  @property
  def score(self):
    cdef int * tot = unique_pointer_to_pointer(deref(self.thisptr).tot, self.N)
    gene_a = self.gene_a
    gene_b = self.gene_b
    return [(a, b, tot[i]) for a, b, i in zip(gene_a, gene_b, range(self.N))]



def _DNetPRO_couples (X, y, percentage=.1, verbose=False, nth=1):

  cdef np.ndarray[float, ndim=2, mode='c'] data = X.astype('float32')
  cdef np.ndarray[int, ndim=1, mode='c'] label = y.astype('int32')

  Nprobe, Nsample = np.shape(X)

  cdef score sc = dnetpro_couples(two_dimension_pointer_for_cython[float, float](&data[0, 0], Nprobe, Nsample),
                                  Nprobe, Nsample,
                                  &label[0],
                                  verbose,
                                  percentage,
                                  nth)

  py_sc = _score()
  py_sc._move(sc)
  return py_sc


