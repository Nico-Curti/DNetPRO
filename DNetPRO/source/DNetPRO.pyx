# distutils: language = c++
# cython: language_level=3

from libcpp.memory cimport unique_ptr
from cpython.ref cimport Py_INCREF
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np

from DNetPRO cimport score, dnetpro_couples, move

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']
__all__ = ['_DNetPRO_couples']

ctypedef np.float32_t FLT32_T
ctypedef np.int32_t INT32_T

np.import_array()

cdef inline np.ndarray _view_1d(object base, void* data, int typenum, Py_ssize_t n):
  cdef np.npy_intp dims[1]
  dims[0] = <np.npy_intp>n
  cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, dims, typenum, data)
  # keep base alive
  Py_INCREF(base)
  np.PyArray_SetBaseObject(arr, base)
  return arr

cdef inline np.ndarray _view_2d(object base, void* data, int typenum, Py_ssize_t n0, Py_ssize_t n1):
  cdef np.npy_intp dims[2]
  dims[0] = <np.npy_intp>n0
  dims[1] = <np.npy_intp>n1
  cdef np.ndarray arr = np.PyArray_SimpleNewFromData(2, dims, typenum, data)
  Py_INCREF(base)
  np.PyArray_SetBaseObject(arr, base)
  return arr

cdef inline np.ndarray _view_3d(object base, void* data, int typenum, Py_ssize_t n0, Py_ssize_t n1, Py_ssize_t n2):
  cdef np.npy_intp dims[3]
  dims[0] = <np.npy_intp>n0
  dims[1] = <np.npy_intp>n1
  dims[2] = <np.npy_intp>n2
  cdef np.ndarray arr = np.PyArray_SimpleNewFromData(3, dims, typenum, data)
  Py_INCREF(base)
  np.PyArray_SetBaseObject(arr, base)
  return arr

cdef class _score:

  cdef void _set_ptr(self, unique_ptr[score] &ptr):
    self.thisptr = move(ptr)
    self.N = self.thisptr.get().N
    self.n_class = self.thisptr.get().n_class

  def __cinit__(self):
    self.N = 0
    self.n_class = 0

  @property
  def score(self):
    return self.tuples_tot()

  @property
  def mcc(self):
    return _view_1d(self, <void*> (<float*> self.thisptr.get().mcc.get()), np.NPY_FLOAT32, self.N)

  @property
  def gene_a(self):
    return _view_1d(self, <void*> (<INT32_T*> self.thisptr.get().gene_a.get()), np.NPY_INT32, self.N)

  @property
  def gene_b(self):
    return _view_1d(self, <void*> (<INT32_T*> self.thisptr.get().gene_b.get()), np.NPY_INT32, self.N)

  @property
  def tot(self):
    return _view_1d(self, <void*> (<INT32_T*> self.thisptr.get().tot.get()), np.NPY_INT32, self.N)

  @property
  def class_score(self):
    return _view_2d(self, <void*> (<INT32_T*> self.thisptr.get().class_score.get()), np.NPY_INT32, self.N, self.n_class)

  def tuples_tot(self):
    cdef int i
    cdef INT32_T* ga = <INT32_T*> self.thisptr.get().gene_a.get()
    cdef INT32_T* gb = <INT32_T*> self.thisptr.get().gene_b.get()
    cdef INT32_T* tt = <INT32_T*> self.thisptr.get().tot.get()
    out = [None] * self.N
    for i in range(self.N):
      out[i] = (ga[i], gb[i], tt[i])
    return out

  def tuples_class_score(self):
    cdef INT32_T i, c
    cdef INT32_T* ga = <INT32_T*> self.thisptr.get().gene_a.get()
    cdef INT32_T* gb = <INT32_T*> self.thisptr.get().gene_b.get()
    cdef INT32_T* cs = <INT32_T*> self.thisptr.get().class_score.get()
    cdef Py_ssize_t K = self.n_class
    out = [None] * self.N
    for i in range(self.N):
      row = [0] * K
      for c in range(K):
        row[c] = cs[i*K + c]
      out[i] = (ga[i], gb[i], tuple(row))
    return out


def _DNetPRO_couples (np.ndarray[FLT32_T, ndim=2, mode='c'] X,
                      np.ndarray[INT32_T, ndim=1, mode='c'] y,
                      float percentage=.1, 
                      bint verbose=False, 
                      int nth=-1):

  """
  Run couples evaluation.
  """
  cdef np.int32_t Nprobe = <np.int32_t> X.shape[0]
  cdef np.int32_t Nsample = <np.int32_t> X.shape[1]

  cdef float* base = <float*> X.data
  cdef float** rowptr = <float**> PyMem_Malloc(Nprobe * sizeof(float*))
  if rowptr == NULL:
    raise MemoryError('Could not allocate row pointer table')

  cdef np.int32_t* labels = <np.int32_t*> y.data
  cdef np.int32_t i
  cdef unique_ptr[score] sc

  try:
    for i in range(Nprobe):
      rowptr[i] = base + i * Nsample

    sc = dnetpro_couples(
      rowptr,
      Nprobe, Nsample,
      labels,
      verbose,
      percentage,
      nth
    )

    py_sc = _score()
    py_sc._set_ptr(sc)
    return py_sc
  finally:
    PyMem_Free(rowptr)
