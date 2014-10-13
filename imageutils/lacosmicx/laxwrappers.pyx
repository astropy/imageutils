# cython: profile=True
"""
laxwrappers.pyx

Author: Curtis McCully
October 2014

This module entirely serves as a wrapper layer between laxutils.c and Python.
"""
import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
np.import_array()
from libc.stdint cimport uint8_t

from cython.parallel import parallel, prange

cdef extern from "laxutils.h":
    float PyMedian(float * a, int n) nogil
    float PyOptMed3(float * a) nogil
    float PyOptMed5(float * a) nogil
    float PyOptMed7(float * a) nogil
    float PyOptMed9(float * a) nogil
    float PyOptMed25(float * a) nogil
    float * PyMedFilt3(float * data, int nx, int ny) nogil
    float * PyMedFilt5(float * data, int nx, int ny) nogil
    float * PyMedFilt7(float * data, int nx, int ny) nogil
    float * PySepMedFilt3(float * data, int nx, int ny) nogil
    float * PySepMedFilt5(float * data, int nx, int ny) nogil
    float * PySepMedFilt7(float * data, int nx, int ny) nogil
    float * PySepMedFilt9(float * data, int nx, int ny) nogil
    float * PySubsample(float * data, int nx, int ny) nogil
    float * PyRebin(float * data, int nx, int ny) nogil
    float * PyConvolve(float * data, float * kernel, int nx, int ny, int kernx,
                      int kerny) nogil
    float * PyLaplaceConvolve(float * data, int nx, int ny) nogil
    uint8_t * PyDilate3(uint8_t * data, int nx, int ny) nogil
    uint8_t * PyDilate5(uint8_t * data, int niter, int nx, int ny) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def median(np.ndarray[np.float32_t, mode='c', cast=True] a, int n):
    """Find the median of an array a of length "n".

    Wrapper for PyMedian in laxutils.
    """
    cdef float * aptr = < float * > np.PyArray_DATA(a)
    cdef float med = 0.0
    with nogil:
        med = PyMedian(aptr, n)
    return med


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed3(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """Optimized method to find the median value of an array "a" of length 3.

    Wrapper for PyOtMed3 in laxutils.
    """
    cdef float * aptr3 = < float * > np.PyArray_DATA(a)
    cdef float med3 = 0.0
    with nogil:
        med3 = PyOptMed3(aptr3)
    return med3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed5(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """Optimized method to find the median value of an array "a" of length 5.

    Wrapper for PyOtMed5 in laxutils.
    """
    cdef float * aptr5 = < float * > np.PyArray_DATA(a)
    cdef float med5 = 0.0
    with nogil:
        med5 = PyOptMed5(aptr5)
    return med5


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed7(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """Optimized method to find the median value of an array "a" of length 7.

    Wrapper for PyOtMed7 in laxutils.
    """
    cdef float * aptr7 = < float * > np.PyArray_DATA(a)
    cdef float med7 = 0.0
    with nogil:
        med7 = PyOptMed7(aptr7)
    return med7


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed9(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """Optimized method to find the median value of an array "a" of length 9.

    Wrapper for PyOtMed9 in laxutils.
    """
    cdef float * aptr9 = < float * > np.PyArray_DATA(a)
    cdef float med9 = 0.0
    with nogil:
        med9 = PyOptMed9(aptr9)
    return med9


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed25(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """Optimized method to find the median value of an array "a" of length 25.

    Wrapper for PyOtMed3 in laxutils.
    """
    cdef float * aptr25 = < float * > np.PyArray_DATA(a)
    cdef float med25 = 0.0
    with nogil:
        med25 = PyOptMed25(aptr25)
    return med25


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d3):
    """Calculate the 3x3 median filter of an array.

    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt3 in laxutils.
    """
    cdef int nx = d3.shape[1]
    cdef int ny = d3.shape[0]
    cdef float * d3ptr = < float * > np.PyArray_DATA(d3)
    cdef float * med3ptr
    with nogil:
        med3ptr = PyMedFilt3(d3ptr, nx, ny)
    cdef np.float32_t[:, ::1] m3_memview = < np.float32_t[:ny, :nx] > med3ptr
    return np.asarray(m3_memview)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d5):
    """Calculate the 5x5 median filter of an array.

    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt5 in laxutils.
    """
    cdef int nx = d5.shape[1]
    cdef int ny = d5.shape[0]
    cdef float * d5ptr = < float * > np.PyArray_DATA(d5)
    cdef float * med5ptr
    with nogil:
        med5ptr = PyMedFilt5(d5ptr, nx, ny)
    cdef np.float32_t[:, ::1] m5_memview = < np.float32_t[:ny, :nx] > med5ptr
    return np.asarray(m5_memview)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d7):
    """ Calculate the 7x7 median filter of an array.

    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt7 in laxutils.
    """
    cdef int nx = d7.shape[1]
    cdef int ny = d7.shape[0]
    cdef float * d7ptr = < float * > np.PyArray_DATA(d7)
    cdef float * med7ptr
    with nogil:
        med7ptr = PyMedFilt7(d7ptr, nx, ny)
    cdef np.float32_t[:, ::1] m7_memview = < np.float32_t[:ny, :nx] > med7ptr
    return np.asarray(m7_memview)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep3):
    """ Calculate the 3x3 separable median filter of an array.

    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt3 in laxutils.
    """
    cdef int nx = dsep3.shape[1]
    cdef int ny = dsep3.shape[0]
    cdef float * dsep3ptr = < float * > np.PyArray_DATA(dsep3)
    cdef float * sepmed3ptr
    with nogil:
        sepmed3ptr = PySepMedFilt3(dsep3ptr, nx, ny)
    cdef np.float32_t[:, ::1] sepm3_mv = < np.float32_t[:ny, :nx] > sepmed3ptr
    return np.asarray(sepm3_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep5):
    """Calculate the 5x5 separable median filter of an array.

    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt5 in laxutils.
    """
    cdef int nx = dsep5.shape[1]
    cdef int ny = dsep5.shape[0]
    cdef float * dsep5ptr = < float * > np.PyArray_DATA(dsep5)
    cdef float * sepmed5ptr
    with nogil:
        sepmed5ptr = PySepMedFilt5(dsep5ptr, nx, ny)
    cdef np.float32_t[:, ::1] m5_mv = < np.float32_t[:ny, :nx] > sepmed5ptr
    return np.asarray(m5_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep7):
    """Calculate the 7x7 separable median filter of an array.

    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt7 in laxutils.
    """
    cdef int nx = dsep7.shape[1]
    cdef int ny = dsep7.shape[0]
    cdef float * dsep7ptr = < float * > np.PyArray_DATA(dsep7)
    cdef float * sepmed7ptr
    with nogil:
        sepmed7ptr = PySepMedFilt7(dsep7ptr, nx, ny)
    cdef np.float32_t[:, ::1] m7_mv = < np.float32_t[:ny, :nx] > sepmed7ptr
    return np.asarray(m7_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep9):
    """ Calculate the 9x9 separable median filter of an array.

    The median filter is not calculated for a 4 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt9 in laxutils.
    """
    cdef int nx = dsep9.shape[1]
    cdef int ny = dsep9.shape[0]
    cdef float * dsep9ptr = < float * > np.PyArray_DATA(dsep9)
    cdef float * sepmed9ptr
    with nogil:
        sepmed9ptr = PySepMedFilt9(dsep9ptr, nx, ny)
    cdef np.float32_t[:, ::1] m9_mv = < np.float32_t[:ny, :nx] > sepmed9ptr
    return np.asarray(m9_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def subsample(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsub):
    """
    Subsample an array 2x2 given an input array dsub. Each pixel is
    replicated into 4 pixels; no averaging is performed. The array needs to be
    C-contiguous order. Wrapper for PySubsample in laxutils.
    """
    cdef int nx = dsub.shape[1]
    cdef int ny = dsub.shape[0]
    cdef int nx2 = 2 * nx
    cdef int ny2 = 2 * ny
    cdef float * dsubptr = < float * > np.PyArray_DATA(dsub)
    cdef float * subsamptr
    with nogil:
        ssptr = PySubsample(dsubptr, nx, ny)
    cdef np.float32_t[:, ::1] subsamp_mv = < np.float32_t[:ny2, :nx2] > ssptr
    return np.asarray(subsamp_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rebin(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] drebin):
    """ Rebin an array 2x2.

    Rebin the array by block averaging 4 pixels back into
    1. This is effectively the opposite of subsample (although subsample does
    not do an average). The array needs to be C-contiguous order. Wrapper for
    PyRebin in laxutils.
    """
    cdef int nx = drebin.shape[1] / 2
    cdef int ny = drebin.shape[0] / 2

    cdef float * drebinptr = < float * > np.PyArray_DATA(drebin)
    cdef float * rebinptr
    with nogil:
        rebinptr = PyRebin(drebinptr, nx, ny)
    cdef np.float32_t[:, ::1] rebin_mv = < np.float32_t[:ny, :nx] > rebinptr
    return np.asarray(rebin_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dconv,
             np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] kernel):
    """Convolve an array with a kernel.

    Both the data and kernel arrays need to be C-contiguous order. Wrapper for
    PyConvolve in laxutils.
    """
    cdef int nx = dconv.shape[1]
    cdef int ny = dconv.shape[0]
    cdef float * dconvptr = < float * > np.PyArray_DATA(dconv)
    cdef int knx = kernel.shape[1]
    cdef int kny = kernel.shape[0]
    cdef float * kernptr = < float * > np.PyArray_DATA(kernel)
    cdef float * convptr
    with nogil:
        convptr = PyConvolve(dconvptr, kernptr, nx, ny, knx, kny)
    cdef np.float32_t[:, ::1] conv_mv = < np.float32_t[:ny, :nx] > convptr
    return np.asarray(conv_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def laplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dl):
    """ Convolve an array with the Laplacian kernel.

    The kernel is as follows:
     0 -1  0
    -1  4 -1
     0 -1  0
    This is a discrete version of the Laplacian operator.
    The array needs to be C-contiguous order. Wrapper for PyLaplaceConvolve
    in laxutils.
    """
    cdef int nx = dl.shape[1]
    cdef int ny = dl.shape[0]
    cdef float * dlapptr = < float * > np.PyArray_DATA(dl)
    cdef float * lapptr
    with nogil:
        lapptr = PyLaplaceConvolve(dlapptr, nx, ny)
    cdef np.float32_t[:, ::1] lap_mv = < np.float32_t[:ny, :nx] > lapptr
    return np.asarray(lap_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dilate3(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] dgrow):
    """ Perform a boolean dilation on an array.

    Dilation is the boolean equivalent of a convolution but using logical ors
    instead of a sum.
    We apply the following kernel:
    1 1 1
    1 1 1
    1 1 1
    The binary dilation is not computed for a 1 pixel border around the image.
    These pixels are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyDilate3 in laxutils.
    """

    cdef int nx = dgrow.shape[1]
    cdef int ny = dgrow.shape[0]
    cdef uint8_t * dgrowptr = < uint8_t * > np.PyArray_DATA(dgrow)
    cdef uint8_t * growptr
    with nogil:
        growptr = PyDilate3(dgrowptr, nx, ny)
    cdef uint8_t[:, ::1] growarr = np.zeros((ny, nx), dtype=np.uint8)
    cdef int i, j = 0
    cdef int nxi = 0
    with nogil:
        for i in prange(ny):
            nxi = nx * i
            for j in range(nx):
                growarr[i, j] = growptr[nxi + j]

    return np.asarray(growarr, dtype=np.bool)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dilate5(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] ddilate,
           int niter):
    """Do niter iterations of boolean dilation on an array.

    Dilation is the boolean equivalent of a convolution but using logical ors
    instead of a sum.
    We apply the following kernel:
    0 1 1 1 0
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    0 1 1 1 0
    The edges are padded with zeros so that the dilation operator is defined
    for all pixels. The array needs to be C-contiguous order. Wrapper for
    PyDilate5 in laxutils.
    """
    cdef int nx = ddilate.shape[1]
    cdef int ny = ddilate.shape[0]
    cdef uint8_t * ddilateptr = < uint8_t * > np.PyArray_DATA(ddilate)
    cdef uint8_t * dilateptr
    with nogil:
        dilateptr = PyDilate5(ddilateptr, niter, nx, ny)
    cdef uint8_t[:, ::1] dilatearr = np.zeros((ny, nx), dtype=np.uint8)
    cdef int i = 0
    cdef int j = 0
    cdef int nxj = 0
    with nogil:
        for j in prange(ny):
            nxj = nx * j
            for i in range(nx):
                dilatearr[j, i] = dilateptr[i + nxj]

    return np.asarray(dilatearr, dtype=np.bool)
