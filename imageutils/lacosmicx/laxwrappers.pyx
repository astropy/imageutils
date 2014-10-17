"""
laxwrappers.pyx

Author: Curtis McCully
October 2014

This module entirely serves as a wrapper layer between laxutils.c and Python.
"""
# cython: profile=True
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
    void PyMedFilt3(float * data, float * output, int nx, int ny) nogil
    void PyMedFilt5(float * data, float * output, int nx, int ny) nogil
    void PyMedFilt7(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt3(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt5(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt7(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt9(float * data, float * output, int nx, int ny) nogil
    void PySubsample(float * data, float * output, int nx, int ny) nogil
    void PyRebin(float * data, float * output, int nx, int ny) nogil
    void PyConvolve(float * data, float * kernel, float * output, int nx,
                    int ny, int kernx, int kerny) nogil
    void PyLaplaceConvolve(float * data, float * output, int nx, int ny) nogil
    void PyDilate3(uint8_t * data, uint8_t * output, int nx, int ny) nogil
    void PyDilate5(uint8_t * data, uint8_t * output, int niter, int nx,
                   int ny) nogil


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def median(np.ndarray[np.float32_t, mode='c', cast=True] a, int n):
    """median(a, n) -> float
    Find the median of the first n elements of an array a. Returns a float.

    Wrapper for PyMedian in laxutils.
    """
    cdef float * aptr = < float * > np.PyArray_DATA(a)
    cdef float med = 0.0
    with nogil:
        med = PyMedian(aptr, n)
    return med


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed3(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed3(a) -> float
    Optimized method to find the median value of an array "a" of length 3.

    Wrapper for PyOtMed3 in laxutils.
    """
    cdef float * aptr3 = < float * > np.PyArray_DATA(a)
    cdef float med3 = 0.0
    with nogil:
        med3 = PyOptMed3(aptr3)
    return med3


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed5(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed5(a) -> float
    Optimized method to find the median value of an array "a" of length 5.

    Wrapper for PyOtMed5 in laxutils.
    """
    cdef float * aptr5 = < float * > np.PyArray_DATA(a)
    cdef float med5 = 0.0
    with nogil:
        med5 = PyOptMed5(aptr5)
    return med5


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed7(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed7(a) -> float
    Optimized method to find the median value of an array "a" of length 7.

    Wrapper for PyOtMed7 in laxutils.
    """
    cdef float * aptr7 = < float * > np.PyArray_DATA(a)
    cdef float med7 = 0.0
    with nogil:
        med7 = PyOptMed7(aptr7)
    return med7


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed9(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed9(a) -> float
    Optimized method to find the median value of an array "a" of length 9.

    Wrapper for PyOtMed9 in laxutils.
    """
    cdef float * aptr9 = < float * > np.PyArray_DATA(a)
    cdef float med9 = 0.0
    with nogil:
        med9 = PyOptMed9(aptr9)
    return med9


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optmed25(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed25(a) -> float
    Optimized method to find the median value of an array "a" of length 25.

    Wrapper for PyOtMed25 in laxutils.
    """
    cdef float * aptr25 = < float * > np.PyArray_DATA(a)
    cdef float med25 = 0.0
    with nogil:
        med25 = PyOptMed25(aptr25)
    return med25


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d3):
    """medfilt3(data) -> array
    Calculate the 3x3 median filter of an array.

    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt3 in laxutils.
    """
    cdef int nx = d3.shape[1]
    cdef int ny = d3.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)
    cdef float * d3ptr = < float * > np.PyArray_DATA(d3)
    cdef float * outd3ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt3(d3ptr, outd3ptr, nx, ny)

    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d5):
    """medfilt5(data) -> array
    Calculate the 5x5 median filter of an array.

    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt5 in laxutils.
    """
    cdef int nx = d5.shape[1]
    cdef int ny = d5.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)
    cdef float * d5ptr = < float * > np.PyArray_DATA(d5)
    cdef float * outd5ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt5(d5ptr, outd5ptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def medfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d7):
    """medfilt7(data) -> array
    Calculate the 7x7 median filter of an array.

    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt7 in laxutils.
    """
    cdef int nx = d7.shape[1]
    cdef int ny = d7.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * d7ptr = < float * > np.PyArray_DATA(d7)
    cdef float * outd7ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt7(d7ptr, outd7ptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep3):
    """sepmedfilt3(data) -> array
    Calculate the 3x3 separable median filter of an array.

    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt3 in laxutils.
    """
    cdef int nx = dsep3.shape[1]
    cdef int ny = dsep3.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep3ptr = < float * > np.PyArray_DATA(dsep3)
    cdef float * outdsep3ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt3(dsep3ptr, outdsep3ptr, nx, ny)
    return np.asarray(output)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep5):
    """sepmedfilt5(data) -> array
    Calculate the 5x5 separable median filter of an array.

    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt5 in laxutils.
    """
    cdef int nx = dsep5.shape[1]
    cdef int ny = dsep5.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep5ptr = < float * > np.PyArray_DATA(dsep5)
    cdef float * outdsep5ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt5(dsep5ptr, outdsep5ptr, nx, ny)

    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep7):
    """sepmedfilt7(data) -> array
    Calculate the 7x7 separable median filter of an array.

    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt7 in laxutils.
    """
    cdef int nx = dsep7.shape[1]
    cdef int ny = dsep7.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep7ptr = < float * > np.PyArray_DATA(dsep7)
    cdef float * outdsep7ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt7(dsep7ptr, outdsep7ptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep9):
    """sepmedfilt9(data) -> array
    Calculate the 9x9 separable median filter of an array.

    The median filter is not calculated for a 4 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt9 in laxutils.
    """
    cdef int nx = dsep9.shape[1]
    cdef int ny = dsep9.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep9ptr = < float * > np.PyArray_DATA(dsep9)
    cdef float * outdsep9ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt9(dsep9ptr, outdsep9ptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def subsample(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsub):
    """subsample(dsub) -> array
    Subsample an array 2x2 given an input array dsub.

    Each pixel is replicated into 4 pixels; no averaging is performed.
    The array needs to be C-contiguous order. Wrapper for PySubsample in
    laxutils.
    """
    cdef int nx = dsub.shape[1]
    cdef int ny = dsub.shape[0]
    cdef int nx2 = 2 * nx
    cdef int ny2 = 2 * ny

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny2, nx2), dtype=np.float32)

    cdef float * dsubptr = < float * > np.PyArray_DATA(dsub)
    cdef float * outdsubptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySubsample(dsubptr, outdsubptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rebin(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] drebin):
    """rebin(data) -> array
    Rebin an array 2x2.

    Rebin the array by block averaging 4 pixels back into
    1. This is effectively the opposite of subsample (although subsample does
    not do an average). The array needs to be C-contiguous order. Wrapper for
    PyRebin in laxutils.
    """
    cdef int nx = drebin.shape[1] / 2
    cdef int ny = drebin.shape[0] / 2

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * drebinptr = < float * > np.PyArray_DATA(drebin)
    cdef float * outdrebinptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyRebin(drebinptr, outdrebinptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dconv,
             np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] kernel):
    """convolve(data, kernel) -> array
    Convolve an array with a kernel.

    Both the data and kernel arrays need to be C-contiguous order. Wrapper for
    PyConvolve in laxutils.
    """
    cdef int nx = dconv.shape[1]
    cdef int ny = dconv.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dconvptr = < float * > np.PyArray_DATA(dconv)
    cdef float * outdconvptr = < float * > np.PyArray_DATA(output)

    cdef int knx = kernel.shape[1]
    cdef int kny = kernel.shape[0]
    cdef float * kernptr = < float * > np.PyArray_DATA(kernel)

    with nogil:
        PyConvolve(dconvptr, kernptr, outdconvptr, nx, ny, knx, kny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def laplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dl):
    """laplaceconvolve(data) -> array
    Convolve an array with the Laplacian kernel.

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

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dlapptr = < float * > np.PyArray_DATA(dl)
    cdef float * outdlapptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyLaplaceConvolve(dlapptr, outdlapptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dilate3(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] dgrow):
    """dilate3(data) -> array
    Perform a boolean dilation on an array.

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

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.bool)

    cdef uint8_t * dgrowptr = < uint8_t * > np.PyArray_DATA(dgrow)
    cdef uint8_t * outdgrowptr = < uint8_t * > np.PyArray_DATA(output)
    with nogil:
        PyDilate3(dgrowptr, outdgrowptr, nx, ny)
    return output


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dilate5(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] ddilate,
           int niter):
    """dilate5(data) -> array
    Do niter iterations of boolean dilation on an array.

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

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.bool)

    cdef uint8_t * ddilateptr = < uint8_t * > np.PyArray_DATA(ddilate)
    cdef uint8_t * outddilateptr = < uint8_t * > np.PyArray_DATA(output)
    with nogil:
        PyDilate5(ddilateptr, outddilateptr, niter, nx, ny)
    return output
