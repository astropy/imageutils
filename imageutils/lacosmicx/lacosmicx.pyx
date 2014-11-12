"""
Name : Lacosmicx
Author : Curtis McCully
Date : October 2014
"""
# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
np.import_array()

from libcpp cimport bool
from laxwrappers import *

from cython.parallel cimport parallel, prange

from libc.stdlib cimport abort, malloc, free

cdef extern from "laxutils.h":
    float PyMedian(float * a, int n) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] indat,
        np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] inmask=None,
        float sigclip=4.5, float sigfrac=0.3, float objlim=5.0, float gain=1.0,
        float readnoise=6.5, float satlevel=65536.0, float pssl=0.0,
        int niter=4, sepmed=True, cleantype='meanmask', fsmode='median',
        psfmodel='gauss', float psffwhm=2.5, int psfsize=7, psfk=None,
        float psfbeta=4.765, verbose=False, retclean=False):
    """run(indat, **kwargs) -> array
    Run the LACosmic algorithm to detect cosmic rays in an array.

    Note: To reproduce the most similar behavior to the original LA Cosmic
    (written in IRAF), set  inmask = None, satlevel = np.inf, sepmed=False,
    cleantype='medmask', and fsmode='median'.

    Returns:
    ========
    If retclean==False, return a cosmic ray mask (boolean) array with values
        of True where there are cosmic ray detections.

    If retclean==True, return the cosmic ray mask and the cleaned data array.

    Arguments:
    ========
    indat: Input data array that will be used for cosmic ray detection.

    Keyword Arguements:
    ===================
    inmask: Input bad pixel mask (boolean array). Values of True will be
        ignored in the cosmic ray detection/cleaning process. Default: None.

    sigclip: Laplacian-to-noise limit for cosmic ray detection. Lower values
        will flag more pixels as cosmic rays. Default: 4.5.

    sigfrac: Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a lapacian-to-noise detection limit of
        sigfrac * sigclip will be used. Default: 0.3.

    objlim: Minimum contrast between Laplacian image and the fine structure
        image. Increase this value if cores of bright stars are flagged as
        cosmic rays. Default: 5.0.

    pssl: Previously subtracted sky level (in ADU). We always need to work in
        electrons for cosmic ray detection, so we need to know the sky level
        that has been subtracted so we can add it back in. Default: 0.0.

    gain: Gain of the image (electrons / ADU). We always need to work in
        electrons for cosmic ray detection. Default: 1.0

    readnoise: Read noise of the image (electrons). This is used in generating
        the noise model of the image. Default: 6.5.

    satlevel: Saturation of level of the image (electrons). This value is used
        to detect saturated stars and pixels at or above this level are added
        to the mask. Default: 65536.0.

    niter: Number of iterations of the LA Cosmic algorithm to perform.
        Default: 4.

    sepmed: Use the separable median filter instead of the full median filter.
        The separable median is not identical to the full median filter, but
        they are approximately the same and the separable median filter is
        significantly faster and still detects cosmic rays well.
        Default: True

    cleantype: Set which clean algorithm is used. The choices are as follows:
        "median": An umasked 5x5 median filter
        "medmask": A masked 5x5 median filter
        "meanmask": A masked 5x5 mean filter
        "idw": A masked 5x5 inverse distance weighted interpolation
        Default: "meanmask".

    fsmode: Method to build the fine structure image. Allowed choices are:
        "median": Use the median filter in the standard LA Cosmic algorithm
        "convolve": Convolve the image with the psf kernel to calculate the
            fine structure image.
        Default: "median".

    psfmodel: Model to use to generate the psf kernel if fsmode == 'convolve'
        psfk is None. Allowed values are "gauss" and "moffat" which generate
        Gaussian and Moffat profiles respectively. Default: "gauss".

    psffwhm: Full Width Half Maximum of the PSF to use to generate
        the kernel. Default: 2.5.

    psfsize: Size of the kernel to calculate. Returned kernel will
        have size psfsize x psfsize. kernsize should be odd. Default: 7.

    psfk: PSF kernel array to use for the fine structure image if
        fsmode == 'convolve'. If None and  fsmode == 'convolve', we calculate
        the psf kernel using the psfmodel. Default: None.

    psfbeta: Moffat beta parameter. Only used if fsmode=='convolve' and
        psfmodel=='moffat'. Default: 4.765.

    verbose: Print to the screen or not (boolean). Default: False.

    retclean: If True, return both the cleaned array and the crmask:
        (crmask, cleanarr). If False, return only the cosmic ray mask (crmask).
        Default: False.
    """

    # Grab the sizes of the input array
    cdef int nx = indat.shape[1]
    cdef int ny = indat.shape[0]

    # Tell the compiler about the loop indices so it can optimize them.
    cdef int i, j = 0

    # Make a copy of the data as the cleanarr that we work on
    # This guarantees that that the data will be contiguous and makes sure we
    # don't edit the input data.
    cleanarr = np.empty((ny, nx), dtype=np.float32)
    # Set the initial values to those of the data array
    cleanarr[:, :] = indat[:, :]

    # Setup the mask
    if inmask is None:
        # By default don't mask anything
        mask = np.zeros((ny, nx), dtype=np.uint8, order='C')
    else:
        # Make a copy of the input mask
        mask = np.empty((ny, nx), dtype=np.uint8, order='C')
        mask[:, :] = inmask[:, :]

    # Add back in the previously subtracted sky level and multiply by the gain
    # The statistics only work properly with electrons.
    cleanarr += pssl
    cleanarr *= gain

    # Find the saturated stars and add them to the mask
    updatemask(np.asarray(cleanarr), np.asarray(mask), satlevel, sepmed)

    # Find the unmasked pixels to calculate the sky.
    gooddata = np.zeros(nx * ny - np.asarray(mask).sum(), dtype=np.float32,
                        order='c')

    igoodpix = 0

    gooddata[:] = cleanarr[np.logical_not(mask)]

    # Get the default background level for large cosmic rays.
    backgroundlevel = median(gooddata, len(gooddata))
    del gooddata

    # Set up the psf kernel if necessary.
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

        if cleantype != 'median':
            del m5

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

        # Update the crmask with the cosmics we have found
        crmask[:, :] = np.logical_or(crmask, cosmics)[:, :]
        del cosmics
        if verbose:
            print "{} cosmic pixels this iteration".format(numcr)

        # If we didn't find anything, we're done.
        if numcr == 0:
            break

        # otherwise clean the image and iterate
        if cleantype == 'median':
        # Unmasked median filter
            cleanarray[crmask] = m5[crmask]
            del m5
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
        output = (crmask.astype(np.bool), cleanarr)
    else:
        del cleanarr
        output = crmask.astype(np.bool)
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def updatemask(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] data,
               np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] mask,
               float satlevel, bool sepmed):
    """updatemask(data, mask, satlevel, sepmed)
     Find staturated stars and puts them in the mask.

     This can then be used to avoid these regions in cosmic detection
     and cleaning procedures. The median filter is used to find large symmetric
     regions of saturated pixels (i.e. saturated stars).

    Arguments:
    =========
    data: The data array in which we look for saturated stars.

    mask: Bad pixel mask (boolean array). This mask will be dilated using
        dilate3 and then combined with the saturated star mask.

    satlevel: Saturation level of the image (float). This value can be lowered
        if the cores of bright (saturated) stars are not being masked.

    sepmed: Use the separable median or not (boolean). The separable median is
        not identical to the full median filter, but they are approximately
        the same and the separable median filter is significantly faster.
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
    del satpixels
    # Combine the saturated pixels with the given input mask
    # Note, we work on the mask pixels in place
    mask[:, :] = np.logical_or(dilsatpixels, grow_mask)[:, :]
    del grow_mask


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void clean_meanmask(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                         bool[:, ::1] mask, int nx, int ny,
                         float backgroundlevel):
    """ clean_meanmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)
    Clean the bad pixels in cleanarr using a 5x5 masked mean filter.

    Arguments:
    ==========
    cleanarr: The array to be cleaned (float array).

    crmask: Cosmic ray mask (boolean array). Pixels with a value of True in
        this mask will be cleaned.

    mask: Bad pixel mask (boolean array)

    nx: size of cleanarr in the x-direction (int). Note cleanarr has dimensions
        ny x nx

    ny: size of cleanarr in the y-direction (int). Note cleanarr has dimensions
        ny x nx

    backgroundlevel: Average value of the background (float). This value will
        be used if there are no good pixels in a 5x5 region.
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
                    # ignoring any pixels that are masked
                    for l in range(-2, 3):
                        for k in range(-2, 3):
                            badpix = crmask[j + l, i + k]
                            badpix = badpix or mask[j + l, i + k]
                            if not badpix:
                                s = s + cleanarr[j + l, i + k]
                                numpix = numpix + 1

                    # if the pixels count is 0
                    # then put in the background of the image
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
                        float backgroundlevel):
    """ clean_medmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)
    Clean the bad pixels in cleanarr using a 5x5 masked median filter.

    Arguments:
    ==========
    cleanarr: The array to be cleaned (float array).

    crmask: Cosmic ray mask (boolean array). Pixels with a value of True in
        this mask will be cleaned.

    mask: Bad pixel mask (boolean array)

    nx: size of cleanarr in the x-direction (int). Note cleanarr has dimensions
        ny x nx

    ny: size of cleanarr in the y-direction (int). Note cleanarr has dimensions
        ny x nx

    backgroundlevel: Average value of the background (float). This value will
        be used if there are no good pixels in a 5x5 region.
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
    """clean_idwinterp(cleanarr, crmask, mask, nx, ny, backgroundlevel)
    Clean the bad pixels in cleanarr using a 5x5 using inverse distance
    weighted interpolation.

    Arguments:
    ==========
    cleanarr: The array to be cleaned (float array).

    crmask: Cosmic ray mask (boolean array). Pixels with a value of True in
        this mask will be cleaned.

    mask: Bad pixel mask (boolean array)

    nx: size of cleanarr in the x-direction (int). Note cleanarr has dimensions
        ny x nx

    ny: size of cleanarr in the y-direction (int). Note cleanarr has dimensions
        ny x nx

    backgroundlevel: Average value of the background (float). This value will
        be used if there are no good pixels in a 5x5 region.
    """

    # Go through all of the pixels, ignore the borders
    cdef int i, j, k, l
    cdef float f11, f12, f21, f22 = backgroundlevel
    cdef int x1, x2, y1, y2
    weightsarr = np.array([[0.35355339, 0.4472136, 0.5, 0.4472136, 0.35355339],
                          [0.4472136, 0.70710678, 1., 0.70710678, 0.4472136],
                          [0.5, 1., 0., 1., 0.5],
                          [0.4472136, 0.70710678, 1., 0.70710678, 0.4472136],
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
    """ gausskernel(psffwhm, kernsize) -> array
    Calculate a Gaussian psf kernel.

    Arguments:
    ==========
    psffwhm (float): Full Width Half Maximum of the PSF to use to generate
        the kernel.

    kernsize (int): Size of the kernel to calculate. Returned kernel will
        have size kernsize x kernsize. kernsize should be odd.
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    # Calculate the offset, r
    r2 = x * x + y * y
    # Calculate the kernel
    sigma2 = psffwhm * psffwhm / 2.35482 / 2.35482
    kernel[:, :] = np.exp(-0.5 * r2 / sigma2)[:, :]
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel

cdef moffatkernel(float psffwhm, float beta, int kernsize):
    """ gausskernel(psffwhm, beta, kernsize) -> array
    Calculate a Moffat psf kernel.

    Arguments:
    ==========
    psffwhm (float): Full Width Half Maximum of the PSF to use to generate
        the kernel.

    beta (float): Moffat beta parameter

    kernsize (int): Size of the kernel to calculate. Returned kernel will
        have size kernsize x kernsize. kernsize should be odd.
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    # Calculate the offset r
    r = (x * x + y * y) ** 0.5
    # Calculate the kernel
    hwhm = psffwhm / 2.0
    alpha = hwhm * (2.0 ** (1.0 / beta) - 1.0) ** 0.5
    kernel[:, :] = ((1.0 + (r / alpha) ** 2.0) ** (-1.0 * beta))[:, :]
    # Normalize the kernel.
    kernel /= kernel.sum()
    return kernel
