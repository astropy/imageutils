# distutils: language = c++
# distutils: sources = functions.cpp
'''
Created on Aug 10, 2014

@author: cmccully
'''
import numpy as np

cimport numpy as np
from libc.math cimport sqrt
cimport cython
np.import_array()

cdef extern from "functions.h":
    float* medfilt3(float* a, int nx, int ny)


def test(d):
    return ctest(d, d.shape[0], d.shape[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ctest(np.ndarray[np.float32_t, ndim=2, mode='c'] d, nrows, ncols):
    cdef float* arrptr = <np.float32_t*> np.PyArray_DATA(d)
    cdef float* m3 = medfilt3(arrptr, nrows, ncols)
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:nrows, :ncols]> m3
    return np.asarray(m3_memview)
