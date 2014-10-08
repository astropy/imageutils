# distutils: language = c++
# distutils: sources = laxutils.cpp
# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
np.import_array()

from cython.parallel import parallel, prange
from libcpp cimport bool

cdef extern from "laxutils.h":
    float _median(float * a, int n) nogil
    float _optmed25(float * a) nogil
    float _optmed3(float * a) nogil
    float _optmed7(float * a) nogil
    float _optmed5(float * a) nogil
    float _optmed9(float * a) nogil
    float * _medfilt3(float * data, int nx, int ny) nogil
    float * _medfilt5(float * data, int nx, int ny) nogil
    float * _medfilt7(float * data, int nx, int ny) nogil
    float * _sepmedfilt3(float * data, int nx, int ny) nogil
    float * _sepmedfilt5(float * data, int nx, int ny) nogil
    float * _sepmedfilt7(float * data, int nx, int ny) nogil
    float * _sepmedfilt9(float * data, int nx, int ny) nogil
    bool * _dilate(bool * data, int iter, int nx, int ny) nogil
    float * _subsample(float * data, int nx, int ny) nogil
    float * _laplaceconvolve(float * data, int nx, int ny) nogil
    float * _rebin(float * data, int nx, int ny) nogil
    bool * _growconvolve(bool * data, int nx, int ny) nogil
    float* _convolve(float* data, float* kernel, int nx, int ny, int kernx, int kerny) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def median(np.ndarray[np.float32_t, mode='c', cast=True] a, int n):
    cdef float * aptr = < float *> np.PyArray_DATA(a)
    cdef float med = 0.0
    with nogil:
        med = _median(aptr, n)
    return med

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed25(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    cdef float * aptr25 = < float *> np.PyArray_DATA(a)
    cdef float med25 = 0.0
    with nogil:
        med25 = _optmed25(aptr25)
    return med25

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed3(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    cdef float * aptr3 = < float *> np.PyArray_DATA(a)
    cdef float med3 = 0.0
    with nogil:
        med3 = _optmed3(aptr3)
    return med3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed5(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    cdef float * aptr5 = < float *> np.PyArray_DATA(a)
    cdef float med5 = 0.0
    with nogil:
        med5 = _optmed5(aptr5)
    return med5

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed7(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    cdef float * aptr7 = < float *> np.PyArray_DATA(a)
    cdef float med7 = 0.0
    with nogil:
        med7 = _optmed7(aptr7)
    return med7

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed9(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    cdef float * aptr9 = < float *> np.PyArray_DATA(a)
    cdef float med9 = 0.0
    with nogil:
        med9 = _optmed9(aptr9)
    return med9

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d3):
    cdef int nx = d3.shape[1]
    cdef int ny = d3.shape[0] 
    cdef float * d3ptr = < float *> np.PyArray_DATA(d3)
    cdef float * med3ptr
    with nogil:
        med3ptr = _medfilt3(d3ptr, nx, ny)
    cdef np.float32_t [:, ::1] m3_memview = < np.float32_t[:ny, :nx] > med3ptr
    return np.asarray(m3_memview)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d5):
    cdef int nx = d5.shape[1]
    cdef int ny = d5.shape[0]
    cdef float * d5ptr = < float *> np.PyArray_DATA(d5)
    cdef float * med5ptr
    with nogil:
        med5ptr = _medfilt5(d5ptr, nx, ny) 
    cdef np.float32_t [:, ::1] m5_memview = < np.float32_t[:ny, :nx] > med5ptr
    return np.asarray(m5_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d7):
    cdef int nx = d7.shape[1]
    cdef int ny = d7.shape[0]
    
    cdef float * d7ptr = < float *> np.PyArray_DATA(d7)
    cdef float * med7ptr
    with nogil:
        med7ptr = _medfilt7(d7ptr, nx, ny)
    cdef np.float32_t [:, ::1] m7_memview = < np.float32_t[:ny, :nx] > med7ptr
    return np.asarray(m7_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep3):
    cdef int nx = dsep3.shape[1]
    cdef int ny = dsep3.shape[0]
    cdef float * dsep3ptr = < float *> np.PyArray_DATA(dsep3)
    cdef float * sepmed3ptr
    with nogil:
        sepmed3ptr = _sepmedfilt3(dsep3ptr, nx, ny)
    cdef np.float32_t [:, ::1] sepm3_memview = < np.float32_t[:ny, :nx] > sepmed3ptr
    return np.asarray(sepm3_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep5):
    cdef int nx = dsep5.shape[1]
    cdef int ny = dsep5.shape[0]
    cdef float * dsep5ptr = < float *> np.PyArray_DATA(dsep5)
    cdef float * sepmed5ptr
    with nogil:
        sepmed5ptr = _sepmedfilt5(dsep5ptr, nx, ny)
    cdef np.float32_t [:, ::1] m5_memview = < np.float32_t[:ny, :nx] > sepmed5ptr
    return np.asarray(m5_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep7):
    cdef int nx = dsep7.shape[1]
    cdef int ny = dsep7.shape[0]
    cdef float * dsep7ptr = < float *> np.PyArray_DATA(dsep7)
    cdef float * sepmed7ptr
    with nogil:
        sepmed7ptr = _sepmedfilt7(dsep7ptr , nx, ny)
    cdef np.float32_t [:, ::1] m7_memview = < np.float32_t[:ny, :nx] > sepmed7ptr
    return np.asarray(m7_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep9):
    cdef int nx = dsep9.shape[1]
    cdef int ny = dsep9.shape[0]
    cdef float * dsep9ptr = < float *> np.PyArray_DATA(dsep9)
    cdef float * sepmed9ptr
    with nogil:
        sepmed9ptr = _sepmedfilt9(dsep9ptr, nx, ny)
    cdef np.float32_t [:, ::1] m9_memview = < np.float32_t[:ny, :nx] > sepmed9ptr
    return np.asarray(m9_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dilate(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] ddilate, int niter):
    cdef int nx = ddilate.shape[1]
    cdef int ny = ddilate.shape[0]
    cdef bool* ddilateptr = < bool *> np.PyArray_DATA(ddilate)
    cdef bool* dilateptr
    with nogil: 
        dilateptr = _dilate(ddilateptr, niter, nx, ny)
    cdef bool[:, ::1] dilatearr = np.zeros((ny, nx), dtype=np.uint8)
    cdef int i = 0
    cdef int j = 0
    cdef int nxj = 0
    with nogil:
        for j in prange(ny):
            nxj = nx * j
            for i in range(nx):
                dilatearr[j, i] = dilateptr[i + nxj]

    return np.asarray(dilatearr, dtype=np.bool)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def growconvolve(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] dgrow):
    cdef int nx = dgrow.shape[1]
    cdef int ny = dgrow.shape[0]
    cdef bool* dgrowptr = < bool *> np.PyArray_DATA(dgrow)
    cdef bool* growptr
    with nogil:
        growptr = _growconvolve(dgrowptr, nx, ny)
    cdef bool[:, ::1] growarr = np.zeros((ny, nx), dtype=np.uint8)
    cdef int i, j = 0
    cdef int nxi = 0
    with nogil:
        for i in prange(ny):
            nxi = nx *i 
            for j in range(nx):
                growarr[i, j] = growptr[nxi + j]
    
    return np.asarray(growarr, dtype=np.bool)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def subsample(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsub):
    cdef int nx = dsub.shape[1]
    cdef int ny = dsub.shape[0]
    cdef float* dsubptr = < float *> np.PyArray_DATA(dsub)
    cdef float* subsamptr
    with nogil:
        subsamptr =  _subsample(dsubptr, nx, ny)
    cdef np.float32_t [:, ::1] subsamp_memview = < np.float32_t[:2 * ny, :2 * nx] > subsamptr
    return np.asarray(subsamp_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def laplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dlap):
    cdef int nx = dlap.shape[1]
    cdef int ny = dlap.shape[0]
    cdef float *dlapptr = < float *> np.PyArray_DATA(dlap)
    cdef float* lapptr
    with nogil:
        lapptr = _laplaceconvolve(dlapptr, nx, ny)
    cdef np.float32_t [:, ::1] lap_memview = < np.float32_t[:ny, :nx] > lapptr
    return np.asarray(lap_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dconv, np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] kernel):
    cdef int nx = dconv.shape[1]
    cdef int ny = dconv.shape[0]
    cdef float* dconvptr = < float *> np.PyArray_DATA(dconv)
    cdef int knx = kernel.shape[1]
    cdef int kny = kernel.shape[0]
    cdef float* kernptr = < float *> np.PyArray_DATA(kernel)
    cdef float* convptr
    with nogil:
        convptr = _convolve(dconvptr, kernptr, nx, ny , knx, kny)
    cdef np.float32_t [:, ::1] conv_memview = < np.float32_t[:ny, :nx] > convptr
    return np.asarray(conv_memview)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rebin(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] drebin):
    cdef int nx = drebin.shape[1]
    cdef int ny = drebin.shape[0]
    cdef float* drebinptr = < float *> np.PyArray_DATA(drebin)
    cdef float* rebinptr
    with nogil:
        rebinptr = _rebin(drebinptr, nx / 2, ny / 2)
    cdef np.float32_t [:, ::1] rebin_memview = < np.float32_t[:ny / 2, :nx / 2] > rebinptr
    return np.asarray(rebin_memview)
