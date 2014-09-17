# distutils: language = c++
# distutils: sources = laxutils.cpp

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
np.import_array()
from scipy import ndimage

from libcpp cimport bool

cdef extern from "laxutils.h":
    float _median(float* a, int n) nogil
    float _optmed25(float* a) nogil
    float _optmed3(float* a) nogil
    float _optmed7(float* a) nogil
    float _optmed5(float* a) nogil
    float _optmed9(float* a) nogil
    float* _medfilt3(float* data, int nx, int ny) nogil
    float* _medfilt5(float* data, int nx, int ny) nogil
    float* _medfilt7(float* data, int nx, int ny) nogil
    float* _sepmedfilt3(float* data, int nx, int ny) nogil
    float* _sepmedfilt5(float* data, int nx, int ny) nogil
    float* _sepmedfilt7(float* data, int nx, int ny) nogil
    float* _sepmedfilt9(float* data, int nx, int ny) nogil
    bool* _dilate(bool* data, int iter, int nx, int ny) nogil
    float* _subsample(float* data, int nx, int ny) nogil
    float* _laplaceconvolve(float* data, int nx, int ny) nogil
    float* _rebin(float* data, int nx, int ny) nogil
    bool* _growconvolve(bool* data, int nx, int ny) nogil

def median(a):
    cdef float* aptr = <float*> np.PyArray_DATA(a)
    cdef int n = len(a)
    cdef float med = 0.0
    with nogil:
        med = _median(aptr,n)
    return med

def optmed25(a):
    cdef float* aptr25 = <float*> np.PyArray_DATA(a)
    cdef float med25 = 0.0
    with nogil:
        med25 =  _optmed25(aptr25)
    return med25

def optmed3(a):
    cdef float* aptr3 = <float*> np.PyArray_DATA(a)
    cdef float med3 = 0.0
    with nogil:
        med3 = _optmed3(aptr3)
    return med3

def optmed5(a):
    cdef float* aptr5 = <float*> np.PyArray_DATA(a)
    cdef float med5 = 0.0
    with nogil:
        med5 = _optmed5(aptr5)
    return med5

def optmed7(a):
    cdef float* aptr7 = <float*> np.PyArray_DATA(a)
    cdef float med7 = 0.0
    with nogil:
        med7 = _optmed7(aptr7)
    return med7

def optmed9(a):
    cdef float* aptr9 = <float*> np.PyArray_DATA(a)
    cdef float med9 = 0.0
    with nogil:
        med9 = _optmed9(aptr9)
    return med9

def medfilt3(d):
    return cymedfilt3(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cymedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:ny, :nx]> _medfilt3(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m3_memview)


def medfilt5(d):
    return cymedfilt5(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cymedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m5_memview = <np.float32_t[:ny, :nx]> _medfilt5(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m5_memview)

def medfilt7(d):
    return cymedfilt7(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cymedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m7_memview
    cdef float* dptr = <float*> np.PyArray_DATA(d)
    cdef float* medptr
    with nogil:
        medptr =  _medfilt7(dptr, nx, ny )
    m7_memview = <np.float32_t[:ny, :nx]> medptr
    return np.asarray(m7_memview)

def sepmedfilt3(d):
    return cysepmedfilt3(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cysepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:ny, :nx]> _sepmedfilt3(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m3_memview)

def sepmedfilt5(d):
    return cysepmedfilt5(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cysepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m5_memview = <np.float32_t[:ny, :nx]> _sepmedfilt5(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m5_memview)

def sepmedfilt7(d):
    return cysepmedfilt7(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cysepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m7_memview = <np.float32_t[:ny, :nx]> _sepmedfilt7(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m7_memview)

def sepmedfilt9(d):
    return cysepmedfilt9(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cysepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m9_memview = <np.float32_t[:ny, :nx]> _sepmedfilt9(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m9_memview)

def dilate(d, niter):
    return cydilate(d,niter,d.shape[1],d.shape[0] )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cydilate(np.ndarray[np.uint8_t, ndim=2, mode='c',cast=True] d, int niter, int nx, int ny):
    dilateptr =   _dilate(<bool*> np.PyArray_DATA(d), niter, nx, ny )
    cdef bool[:,::1] dilatearr = np.zeros((ny,nx), dtype = np.uint8)
    cdef int i,j = 0
    for i in range(ny):
        for j in range(nx):
            dilatearr[i,j] = dilateptr[i*nx + j]
    #cdef bool[:, ::1] dilate_memview = <np.uint8_t[:ny, :nx]>
    return np.asarray(dilatearr,dtype=np.bool)

def growconvolve(d):
    return cygrowconvolve(d,d.shape[1],d.shape[0] )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cygrowconvolve(np.ndarray[np.uint8_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    growptr =   _growconvolve(<bool*> np.PyArray_DATA(d), nx, ny )
    cdef bool[:,::1] growarr = np.zeros((ny,nx), dtype = np.uint8)
    cdef int i,j = 0
    for i in range(ny):
        for j in range(nx):
            growarr[i,j] = growptr[i*nx + j]
    #cdef bool[:, ::1] dilate_memview = <np.uint8_t[:ny, :nx]>
    return np.asarray(growarr,dtype=np.bool)

def subsample(d):
    return cysubsample(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cysubsample(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] subsamp_memview = <np.float32_t[:2*ny, :2*nx]> _subsample(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(subsamp_memview)

def laplaceconvolve(d):
    return cylaplaceconvolve(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cylaplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] lap_memview = <np.float32_t[:ny, :nx]> _laplaceconvolve(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(lap_memview)

def rebin(d):
    return cyrebin(d,d.shape[1],d.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cyrebin(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] rebin_memview = <np.float32_t[:ny/2, :nx/2]> _rebin(<float*> np.PyArray_DATA(d),  nx/2, ny/2 )
    return np.asarray(rebin_memview)
