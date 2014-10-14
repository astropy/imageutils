# cython: profile=True
"""
Name : Lacosmicx
Author : Curtis McCully
Date : October 2014

 About
 =====

 Lacosmicx is designed to detect cosmic rays in images (numpy arrays),
 based on Pieter van Dokkum's L.A.Cosmic algorithm.

 Much of this was originally adapted from cosmics.py written by Malte Tewes.
 I have ported all of the slow functions to Cython/C, and optimized
 where I can. This is designed to be as fast as possible so some of the
 readability has been sacrificed, specifically in the C code.

 L.A.Cosmic = LAplacian Cosmic ray detection

 U{http://www.astro.yale.edu/dokkum/lacosmic/}

 (article : U{http://arxiv.org/abs/astro-ph/0108003})

 This code requires Cython, preferably version >0.21.

 Parallelization is achieved using OpenMP. This code should compile (although
 the Cython files may have issues) using a compiler that does not support OMP,
 e.g. clang.

 Differences from original LACosmic
 ===================

 - Automatic recognition of saturated stars, including their trails.
 This avoids treating such stars as large cosmic rays.

 -I have tried to optimize all of the code as much as possible while
 maintaining the integrity of the algorithm. One of the key speedups is to
 use a separable median filter instead of the true median filter. While these
 are not identical, they produce comparable results and the separable version
 is much faster.

 -This implementation is much faster than the Python by as much as a factor of
 17 depending on the given parameters, even without running multiple threads.
 With multiple threads, this can be increased easily by another factor of 2.
 This implementation is much faster than the original IRAF version
 (orders of magnitude).

 -The arrays always must be C-contiguous, thus all loops are y outer, x inner.
 Note that this follows the Pyfits convention.
 """

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
np.import_array()

from libcpp cimport bool
from laxwrappers import *

from cython.parallel cimport parallel, prange
cimport openmp

from libc.stdlib cimport abort, malloc, free

cdef extern from "laxutils.h":
    float PyMedian(float * a, int n) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] indat,
        np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] inmask=None,
        float sigclip=4.5, float sigfrac=0.3, float objlim=5.0,
        float gain=1.0, float readnoise=6.5, float satlevel=65536.0, float pssl = 0.0,
        int niter=4, sepmed=True, cleantype='meanmask',
        fsmode='median', psfmodel='gauss',
        psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765, bool verbose = False, int nthreads = 4, bool retclean = False):
    """ Run the LACosmic algorithm to detect cosmic rays on an array.

    Keywords:
    ========
    indat
    
    inmask
    
    sigclip
    
    sigfrac
    
    objlim
    
    pssl
    
    gain
    
    readnoise
    
    satlevel
    
    niter
    
    sepmed
    
    cleantype
    
    fsmode
    
    psfmodel
    
    psffwhm
    
    psfsize
    
    psfk
    
    psfbeta
    
    verbose
    
    nthreads
    
     sigclip : increase this if you detect cosmics where there are none.
               Default is 5.0, a good value for earth-bound images.
     objlim : increase this if normal stars are detected as cosmics.
              Default is 5.0, a good value for earth-bound images.
    
     sigclip : laplacian-to-noise limit for cosmic ray detection
     objlim : minimum contrast between laplacian image and fine structure image.
              Use 5.0 if your image is undersampled, HST, ...
    
     satlevel : if we find an agglomeration of pixels above this level,
                we consider it to be a saturated star and do not try to correct
                and pixels around it.This is given in electrons
    
     pssl is the previously subtracted sky level !
    
     real   gain    = 1.0         # gain (electrons/ADU)
     real   readn   = 6.5              # read noise (electrons)
     real   skyval  = 0.           # sky level that has been subtracted (ADU)
     real   sigclip = 4.5          # detection limit for cosmic rays (sigma)
     real   sigfrac = 0.3       # fractional detection limit for neighboring pixels
     real   objlim  = 5.0        # contrast limit between CR and underlying object
     int    niter   = 4         # maximum number of iterations
    
    """
    #Set the number of threads to use for parallel calculations
    openmp.omp_set_dynamic(1)
    openmp.omp_set_num_threads(4)

    #Grab the sizes of the input array
    cdef int nx = indat.shape[1]
    cdef int ny = indat.shape[0]

    #Tell the compiler about the loop indices so it can optimize them.
    cdef int i, j = 0

    # Make a copy of the data as the cleanarr that we work on
    # This guaruntees that that the data will be contiguous and makes sure we
    # don't edit the input data.
    cleanarr = np.empty_like(indat)
    # Set the initial values to those of the data array
    cleanarr[:, :] = indat[:, :]

    #Setup the mask
    if inmask is None:
        # By default don't mask anything
        mask = np.zeros((ny, nx), dtype=np.uint8, order='C')
    else:
        # Make a copy of the input mask
        mask = np.empty_like(inmask, dtype=np.uint8)
        mask[:, :] = inmask[:, :]

    # Find the saturated stars and add them to the mask
    updatemask(np.asarray(cleanarr), np.asarray(mask), satlevel, sepmed)

    #Find the unmasked pixels to calculate the sky.
    gooddata = np.zeros(nx * ny - np.asarray(mask).sum(), dtype=np.float32,
                        order='c')

    igoodpix = 0

    # Add back in the previously subtracted sky level and multiply by the gain
    # The statistics only work properly with electrons.
    cleanarr += pssl
    cleanarr *= gain

    gooddata[:] = cleanarr[np.logical_not(mask)]

    # Get the default background level for large cosmic rays.
    backgroundlevel = median(gooddata, len(gooddata))

    #Set up the psf kernel if necessary.
    if psfk is None and fsmode == 'convolve':
        # calculate the psf kernel psfk
        if psfmodel == 'gauss':
            psfk = gausskernel(psffwhm, psfsize)
        elif psfmodel == 'moffat':
            psfk = moffatkernel(psffwhm, psfbeta, psfsize)
        else:
            raise ValueError('Please choose a supported PSF model.')

    # Define a cosmic ray mask
    # This is what will be returned at the end
    crmask = np.zeros((ny, nx), dtype=np.uint8, order='C')

    # Calculate the detection limit for neighbor pixels
    cdef float sigcliplow = sigfrac * sigclip

    # Run lacosmic for up to maxiter iterations
    # We stop if no more cosmic ray pixels are found (quite rare)
    if verbose:
        print "Starting {} L.A.Cosmic iterations".format(niter)
    for i in range(niter):
        if verbose:
            print "Iteration {}:".format(i + 1)

        # Detect the cosmic rays

        # We subsample, convolve, clip negative values,
        # and rebin to original size
        subsam = subsample(cleanarr)

        conved = laplaceconvolve(subsam)
        del subsam

        conved[conved < 0] = 0.0
        # This is called L+ in the original LA Cosmic/cosmics.py
        s = rebin(conved)
        del conved

        # Build a the noise map, to compare the laplacian to
        if sepmed:
            m5 = sepmedfilt7(cleanarr)
        else:
            m5 = medfilt5(cleanarr)

        # Clip noise so that we can take a square root
        m5[m5 < 0.00001] = 0.00001
        noise = np.sqrt(m5 + readnoise * readnoise)

        # Laplacian signal to noise ratio :
        s /= 2.0 * noise
        # the 2.0 is from the 2x2 subsampling
        # This s is called sigmap in the original lacosmic.cl

        if sepmed:
            sp = sepmedfilt7(s)
        else:
            sp = medfilt5(s)

        # Remove the large structures (s prime) :
        sp = s - sp
        del s

        # Build the fine structure image :
        if fsmode == 'convolve':
            f = convolve(cleanarr, psfk)
        elif fsmode == 'median':
            if sepmed:
                f = sepmedfilt5(cleanarr)
            else:
                f = medfilt3(cleanarr)
        else:
            raise ValueError('Please choose a valid fine structure mode.')

        if sepmed:
            m7 = sepmedfilt9(f)
        else:
            m7 = medfilt7(f)

        f = (f - m7) / noise
        # Clip f as we will divide by f. Similar to the IRAF version.
        f[f < 0.01] = 0.01

        del m7
        del noise

        # Find the candidate cosmic rays
        goodpix = np.logical_not(mask)
        cosmics = np.logical_and(sp > sigclip, goodpix)
        # Note the sp/f and not lplus/f due to the f = f/noise above.
        cosmics = np.logical_and(cosmics, (sp / f) > objlim)
        del f

        # What follows is a special treatment for neighbors, with more relaxed
        # constraints.
        # We grow these cosmics a first time to determine the immediate
        # neighborhood.
        cosmics = dilate3(cosmics)
        cosmics = np.logical_and(cosmics, goodpix)
        # From this grown set, we keep those that have sp > sigmalim
        cosmics = np.logical_and(sp > sigclip, cosmics)

        # Now we repeat this procedure, but lower the detection limit to siglow
        cosmics = dilate3(cosmics)
        cosmics = np.logical_and(cosmics, goodpix)

        del goodpix
        cosmics = np.logical_and(sp > sigcliplow, cosmics)
        del sp

        # Our CR counter
        numcr = cosmics.sum()

        #Update the crmask with the cosmics we have found
        crmask[:, :] = np.logical_or(crmask, cosmics)[:, :]

        if verbose:
            print "{} cosmic pixels this iteration".format(numcr)

        # If we didn't find anything, we're done.
        if numcr == 0:
            break

        # otherwise clean the image and iterate
        if cleantype == 'median':
        # Unmasked median filter
            cleanarray[crmask] = m5[crmask]
        # Masked mean filter
        elif cleantype == 'meanmask':
            clean_meanmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)
        # Masked median filter
        elif cleantype == 'medmask':
            clean_medmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)
        # Inverse distance weighted interpolation
        elif cleantype == 'idw':
            clean_idwinterp(cleanarr, crmask, mask, nx, ny, backgroundlevel)
        else:
            raise ValueError("""cleantype must be one of the following values:
                            [median, meanmask, medmask, idw]""")

    if retclean:
        output = (crmask, cleanarr)
    else:
        del cleanarr
        output = crmask
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def updatemask(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] data,
               np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] mask,
               float satlevel, bool sepmed):
    """
     Uses the satlevel to find saturated stars (not cosmic rays hopefully),
     and puts the result in the mask.
     This can then be used to avoid these regions in cosmic detection
     and cleaning procedures.
     
    Keywords:
    ========
    
    data
    
    mask
    
    satlevel
    
    sepmed
    """

    # Find all of the saturated pixels
    satpixels = data >= satlevel

    # Use the median filter to estimate the large scale structure
    if sepmed:
        m5 = sepmedfilt7(data)
    else:
        m5 = medfilt5(data)

    # Use the median filtered image to find the cores of saturated stars
    # The 10 here is arbitray. Malte Tewes uses 2.0 in cosmics.py, but I
    # wanted to get more of the cores of saturated stars.
    satpixels = np.logical_and(satpixels, m5 > (satlevel / 10.0))

    # Grow the input mask by one pixel to make sure we cover bad pixels
    grow_mask = dilate3(mask)

    # Dilate the saturated star mask to remove edge effects in the mask
    dilsatpixels = dilate5(satpixels, 2)

    # Combine the saturated pixels with the given input mask
    # Note, we work on the mask pixels in place
    mask[:, :] = np.logical_or(dilsatpixels, grow_mask)[:, :]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void clean_meanmask(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                         bool[:, ::1] mask, int nx, int ny,
                         float backgroundlevel):
    """ Clean the bad pixels in cleanarr using a masked mean filter.
    
    """
    # Go through all of the pixels, ignore the borders
    cdef int i, j, k, l, numpix
    cdef float s
    cdef bool badpix

    with nogil, parallel():
        # For each pixel
        for j in prange(2, ny - 2):
            for i in range(2, nx - 2):
                # if the pixel is in the crmask
                if crmask[j, i]:
                    numpix = 0
                    s = 0.0

                    # sum the 25 pixels around the pixel
                    #ignoring any pixels that are masked
                    for l in range(-2, 3):
                        for k in range(-2, 3):
                            badpix = crmask[j + l, i + k]
                            badpix = badpix or mask[j + l, i + k]
                            if not badpix:
                                s = s + cleanarr[j + l, i + k]
                                numpix = numpix + 1

                    # if the pixels count is 0
                    #then put in the background of the image
                    if numpix == 0:
                        s = backgroundlevel
                    else:
                        # else take the mean
                        s = s / float(numpix)

                    cleanarr[j, i] = s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void clean_medmask(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                        bool[:, ::1] mask, int nx, int ny,
                        float backgroundlevel) nogil:
    """ Clean the bad pixels in cleanarr using a masked median filter.
    
    """
    # Go through all of the pixels, ignore the borders
    cdef int k, l, i, j, numpix
    cdef float * medarr
    cdef bool badpixel

    # For each pixel
    with nogil, parallel():
        medarr = < float * > malloc(25 * sizeof(float))
        for j in prange(2, ny - 2):
            for i in range(2, nx - 2):
                # if the pixel is in the crmask
                if crmask[j, i]:
                    numpix = 0
                    # median the 25 pixels around the pixel ignoring
                    # any pixels that are masked
                    for l in range(-2, 3):
                        for k in range(-2, 3):
                            badpixel = crmask[j + l, i + k]
                            badpixel = badpixel or mask[j + l, i + k]
                            if not badpixel:
                                medarr[numpix] = cleanarr[j + l, i + k]
                                numpix = numpix + 1

                    # if the pixels count is 0 then put in the background
                    # of the image
                    if numpix == 0:
                        cleanarr[j, i] = backgroundlevel
                    else:
                        # else take the mean
                        cleanarr[j, i] = PyMedian(medarr, numpix)
        free(medarr)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void clean_idwinterp(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                          bool[:, ::1] mask, int nx, int ny,
                          float backgroundlevel):
    """ Clean the bad pixels in cleanarr using inverse distance weighted
    interpolation.
    
    
    
    """

    # Go through all of the pixels, ignore the borders
    cdef int i, j, k, l
    cdef float f11, f12, f21, f22 = backgroundlevel
    cdef int x1, x2, y1, y2
    weightsarr = np.array([[0.35355339, 0.4472136, 0.5, 0.4472136, 0.35355339],
                          [0.4472136,  0.70710678, 1., 0.70710678, 0.4472136],
                          [0.5,         1.,         0., 1.,         0.5],
                          [0.4472136,   0.70710678, 1., 0.70710678, 0.4472136],
                          [0.35355339, 0.4472136, 0.5, 0.4472136, 0.35355339]],
                          dtype=np.float32)
    cdef float[:, ::1] weights = weightsarr
    cdef float wsum
    cdef float val
    cdef int x, y
    # For each pixel
    with nogil, parallel():

        for j in prange(2, ny - 2):
            for i in range(2, nx - 2):
                # if the pixel is in the crmask
                if crmask[j, i]:
                    wsum = 0.0
                    val = 0.0
                    for l in range(-2, 3):
                        y = j + l
                        for k in range(-2, 3):
                            x = i + k
                            if not (crmask[y, x] or mask[y, x]):
                                val = val + weights[l, k] * cleanarr[y, x]
                                wsum = wsum + weights[l, k]
                    if wsum < 1e-6:
                        cleanarr[j, i] = backgroundlevel
                    else:
                        cleanarr[j, i] = val / wsum


def gausskernel(float psffwhm, int kernsize):
    """ Calculate a gaussian kernel.
    # Assume size is odd and is the side length of the Kernel
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    #Calculate the offset, r
    r2 = x * x + y * y
    #Calculate the kernel
    sigma2 = psffwhm * psffwhm / 2.35482 / 2.35482
    kernel[:, :] = np.exp(-0.5 * r2 / sigma2)[:, :]
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel

cdef moffatkernel(float psffwhm, float beta, int kernsize):
    """ Calculate a moffat kernel.
    # Assume size is odd and is the side length of the Kernel
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    #Calculate the offset r
    r = (x * x + y * y) ** 0.5
    #Calculate the kernel
    hwhm = psffwhm / 2.0
    alpha = hwhm * (2.0 ** (1.0 / beta) - 1.0) ** 0.5
    kernel[:, :] = ((1.0 + (r / alpha) ** 2.0) ** (-1.0 * beta))[:, :]
    # Normalize the kernel.
    kernel /= kernel.sum()
    return kernel
