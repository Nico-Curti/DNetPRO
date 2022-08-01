# distutils: language = c++
# cython: language_level=2

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
cimport numpy as np

from DNetPRO cimport score
from DNetPRO cimport _score
from DNetPRO cimport dnetpro_couples
from DNetPRO cimport move

from misc cimport unique_pointer_to_pointer
from misc cimport two_dimension_pointer_for_cython

import numpy as np

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']
__all__ = ['_DNetPRO_couples']

ctypedef np.float32_t FLT32_T
ctypedef np.int32_t INT32_T


cdef class _score:

  def __init__ (self, INT32_T N=0, INT32_T n_class=0):

    if N != 0 and n_class != 0:

      self.N, self.n_class = (N, n_class)
      self.thisptr.reset(new score(self.N, self.n_class))

    else:

      self.N, self.n_class = (0, 0)
      self.thisptr.reset(new score())

  def __str__ (self):
    return '<Score: N={} n_class={}>'.format(deref(self.thisptr).N, deref(self.thisptr).n_class)


  @property
  def mcc (self):
    cdef float * mcc = unique_pointer_to_pointer(deref(self.thisptr).mcc, self.N)
    return [mcc[i] for i in range(self.N)]

  @property
  def gene_a(self):
    cdef INT32_T * gene_a = unique_pointer_to_pointer(deref(self.thisptr).gene_a, self.N)
    return [gene_a[i] for i in range(self.N)]

  @property
  def gene_b(self):
    cdef INT32_T * gene_b = unique_pointer_to_pointer(deref(self.thisptr).gene_b, self.N)
    return [gene_b[i] for i in range(self.N)]

  @property
  def score(self):
    cdef INT32_T * tot = unique_pointer_to_pointer(deref(self.thisptr).tot, self.N)
    gene_a = self.gene_a
    gene_b = self.gene_b
    return [(a, b, tot[i]) for a, b, i in zip(gene_a, gene_b, range(self.N))]



def _DNetPRO_couples (np.ndarray[FLT32_T, ndim=2, mode='c'] X,
  np.ndarray[INT32_T, ndim=1, mode='c'] y,
  float percentage=.1, INT32_T verbose=0, INT32_T nth=1):

  Nprobe, Nsample = np.shape(X)

  cdef score sc = dnetpro_couples(two_dimension_pointer_for_cython[float, float](&X[0, 0], Nprobe, Nsample),
                                  Nprobe, Nsample,
                                  &y[0],
                                  verbose,
                                  percentage,
                                  nth)

  py_sc = _score()
  py_sc._move(sc)
  return py_sc
