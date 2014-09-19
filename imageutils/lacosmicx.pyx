# distutils: language = c++
# distutils: sources = laxutils.cpp
# cython: profile=True
"""============================================================================
// Name        : Lacosmicx
// Author      : Curtis McCully
// Version     :
// Copyright   :
// Description : Lacosmic Written in C++
============================================================================
"""

"""
 About
 =====

 lacosmicx is designed to detect and clean cosmic ray hits on images (numpy arrays or FITS), using scipy, and based on Pieter van Dokkum's L.A.Cosmic algorithm.

 Most of this code was directly adapted from cosmics.py written by Malte Tewes. I have removed some of the extras that he wrote, ported everything to c++, and optimized any places that I can.
 This is designed to be as fast as possible so some of the readability has been sacrificed.

 L.A.Cosmic = Laplacian cosmic ray detection

 U{http://www.astro.yale.edu/dokkum/lacosmic/}

 (article : U{http://arxiv.org/abs/astro-ph/0108003})


 Differences from original LA-cosmic
 ===================

 - Automatic recognition of saturated stars, including their full saturation trails.
 This avoids that such stars are treated as big cosmics.
 Indeed saturated stars tend to get even uglier when you try to clean them. Plus they
 keep L.A.Cosmic iterations going on forever.
 This feature is mainly for pretty-image production. It is optional, requires one more parameter (a CCD saturation level in ADU), and uses some
 nicely robust morphology operations and object extraction.


 -I have tried to optimize all of the code as much as possible while maintaining the integrity of the algorithm.

 -This implementation is much faster than the Python or Iraf versions by ~factor of 7.

 - In pyfits, data are striped along x dimen, thus all loops
 are y outer, x inner.  or at least they should be...

 sigclip : increase this if you detect cosmics where there are none. Default is 5.0, a good value for earth-bound images.
 objlim : increase this if normal stars are detected as cosmics. Default is 5.0, a good value for earth-bound images.

 Constructor of the cosmic class, takes a 2D numpy array of your image as main argument.
 sigclip : laplacian-to-noise limit for cosmic ray detection
 objlim : minimum contrast between laplacian image and fine structure image. Use 5.0 if your image is undersampled, HST, ...

 satlevel : if we find an agglomeration of pixels above this level, we consider it to be a saturated star and
 do not try to correct and pixels around it.This is given in electrons

 pssl is the previously subtracted sky level !

 real   gain    = 1.0         # gain (electrons/ADU)
 real   readn   = 6.5              # read noise (electrons)
 real   skyval  = 0.           # sky level that has been subtracted (ADU)
 real   sigclip = 4.5          # detection limit for cosmic rays (sigma)
 real   sigfrac = 0.3          # fractional detection limit for neighboring pixels
 real   objlim  = 5.0           # contrast limit between CR and underlying object
 int    niter   = 4            # maximum number of iterations

 */

 """

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
np.import_array()

from libcpp cimport bool
from laxwrappers import *

from cython.parallel import parallel, prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] indat,
                      np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] inmask=None,
                       float sigclip=4.5, float sigfrac=0.3,
                       float objlim=5.0, float readnoise=6.5, float satlevel=65536.0,
                       float pssl=0.0, float gain=1.0 ,
                       int niter=4, fullmedian=False):
    cdef int nx = indat.shape[1]
    cdef int ny = indat.shape[0]
    cdef int i, j = 0

    # Make a copy of the data as the cleanarray that we work on
    cleanarr = np.empty_like(indat)
    cleanarr[:, :] = indat[:, :] 
    
    # The data mask needs to be indexed with (i,j) -> (nx *j+i) (c-style indexed)
    if inmask is None:
        # By default don't mask anything
        mask = np.zeros((ny, nx), dtype=np.uint8, order='C')
    else:
        # Make a copy of the input mask
        mask = np.empty_like(inmask, dtype=np.uint8)
        mask[:, :] = inmask[:, :]
         
    # Find the saturated stars and add them to the mask
    updatemask(np.asarray(cleanarr), np.asarray(mask), satlevel, fullmedian)
    
    gooddata = np.zeros(nx * ny - np.asarray(mask).sum(), dtype=np.float32, order='C')

    igoodpix = 0

    # note the c-style convention here with y as the slow direction and x and the fast direction
    # This follows the pyfits convention as well
    cleanarr += pssl
    cleanarr *= gain

    gooddata[:] = cleanarr[np.logical_not(mask)]
    # Get the default background level for large cosmics
    backgroundlevel = median(gooddata)
    del gooddata

    # Defined a cleaned array
    # This is what we work with in lacosmic iteration
    # Set the initial values to those of the data array

    # Define a cosmic ray mask
    # This is what will be returned at the end
    cdef bool[:, ::1] crmask = np.zeros((ny, nx), dtype=np.uint8, order='C')

    # Run lacosmic for up to maxiter iterations
    # We stop if no more cosmics are found

    print "Starting {} L.A.Cosmic iterations".format(niter)
    for i in range(niter):
        print "Iteration {}:".format(i + 1)

        # Detect the cosmic rays
        ncrpix = lacosmiciteration(np.asarray(cleanarr), np.asarray(mask), np.asarray(crmask),
                                    sigclip, objlim, sigfrac, backgroundlevel, readnoise, nx, ny, fullmedian) 

        print "{} cosmic pixels this iteration".format(ncrpix)
        
        # If we didn't find anything, we're done.
        if ncrpix == 0:
            break
        # otherwise clean the image and iterate

        # Easy way without masking would be :
        # self.cleanarray[crmask] = m5[crmask]
        # Go through and clean the image using a masked mean filter, we outsource this to the clean method
        clean(cleanarr, crmask, nx, ny, backgroundlevel);

    return np.asarray(crmask)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void updatemask(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] data, np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] mask, float satlevel, bool fullmedian):
    """
     Uses the satlevel to find saturated stars (not cosmics !), and puts the result as a mask in the mask.
     This can then be used to avoid these regions in cosmic detection and cleaning procedures.
    """

    # DETECTION
    # Find all of the saturated pixels

    satpixels = data >= satlevel

    # in an attempt to avoid saturated cosmic rays we try prune the saturated stars using the large scale structure
    if fullmedian:
        m5 = medfilt5(data)
    else: 
        m5 = sepmedfilt7(data)
    
    # This mask will include saturated pixels and masked pixels

    satpixels = np.logical_and(satpixels, m5 > (satlevel / 10.0))

    # BUILDING THE MASK

    # Combine the saturated pixels with the given input mask
    # Grow the input mask by one pixel to make sure we cover bad pixels
    grow_mask = growconvolve(mask);

    # We want to dilate both the mask and the saturated stars to remove false detections along the edges of the mask
    dilsatpixels = dilate(satpixels, 2);

    mask = np.logical_or(dilsatpixels, grow_mask)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef clean(float[:, ::1] cleanarr, bool[:, ::1] crmask, int nx, int ny,
        float backgroundlevel):
    # Go through all of the pixels, ignore the borders
    cdef int i, j, nxj, idx, k, l, nxl, numpix
    cdef float s = 0.0

    # For each pixel
    for j in range(2, ny - 2):
        nxj = nx * j;
        for i in range(2, nx - 2):
            # if the pixel is in the crmask
            if crmask[j, i]:
                numpix = 0;
                s = 0.0;
                # sum the 25 pixels around the pixel ignoring any pixels that are masked

                for l in range(-2, 3):
                    nxl = nx * l;
                    for k in range(-2, 3):
                        if not crmask[j + l, i + k]:
                            s += cleanarr[j + l, i + k]
                            numpix += 1

                # if the pixels count is 0 then put in the background of the image
                if numpix == 0:
                    s = backgroundlevel
                else:
                    # else take the mean
                    s /= float(numpix)

                cleanarr[j, i] = s

cdef int lacosmiciteration(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] cleanarr, np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] mask, np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] crmask, float sigclip,
        float objlim, float sigfrac, float backgroundlevel, float readnoise,
        int nx, int ny, bool fullmedian):
    """
     Performs one iteration of the L.A.Cosmic algorithm.
     It operates on cleanarray, and afterwards updates crmask by adding the newly detected
     cosmics to the existing crmask. Cleaning is not done automatically ! You have to call
     clean() after each iteration.
     This way you can run it several times in a row to to L.A.Cosmic "iterations".
     See function lacosmic, that mimics the full iterative L.A.Cosmic algorithm.

     Returns numcr : the number of cosmic pixels detected in this iteration

    """
    # Calculate the sigma value for neighbor pixels
    cdef float sigcliplow = sigfrac * sigclip 
    # We subsample, convolve, clip negative values, and rebin to original size
    subsam = subsample(cleanarr)

    conved = laplaceconvolve(subsam)
    del subsam;

    conved[conved < 0] = 0.0
    s = rebin(conved);
    del conved

    # We build a custom noise map, to compare the laplacian to
    if fullmedian:
        noise = medfilt5(cleanarr);
    else:
        noise = sepmedfilt7(cleanarr);

    # We clip noise so that we can take a square root
    noise[noise < 0.0001] = 0.0001
    noise = np.sqrt(noise + readnoise * readnoise)


    # Laplacian signal to noise ratio :
    s /= 2.0 * noise
    # the 2.0 is from the 2x2 subsampling
    # This s is called sigmap in the original lacosmic.cl

    if fullmedian:
        sp = medfilt5(s)
    else:
        sp = sepmedfilt7(s)

    # We remove the large structures (s prime) :
    sp = s - sp
    del s

    # We build the fine structure image :
    if fullmedian:
        m3 = medfilt3(cleanarr)
        f = medfilt7(m3)
    else:
        m3 = sepmedfilt5(cleanarr)
        f = sepmedfilt9(m3)

    f = (m3 - f) / noise
    f[f < 0.01 ] = 0.01
    # as we will divide by f. like in the iraf version.

    del m3
    del noise

    """//Comments from Malte Tewes
    // In the article that's it, but in lacosmic.cl f is divided by the noise...
    // Ok I understand why, it depends on if you use sp/f or L+/f as criterion.
    // There are some differences between the article and the iraf implementation.
    // So we will stick to the iraf implementation.

    // Now we have our better selection of cosmics :
    //Note the sp/f and not lplus/f ... due to the f = f/noise above.
    """

    cosmics = np.logical_and(np.logical_and(sp > sigclip, np.logical_not(mask)) , (sp / f) > objlim)
    del f
    
    # What follows is a special treatment for neighbors, with more relaxed constraints.
    # We grow these cosmics a first time to determine the immediate neighborhood  :
    cosmics = growconvolve(cosmics)

    # From this grown set, we keep those that have sp > sigmalim
    # so obviously not requiring sp/f > objlim, otherwise it would be pointless
    # This step still feels pointless to me, but we leave it in because the iraf implementation has it
    cosmics = np.logical_and(sp > sigclip, np.logical_and(cosmics, np.logical_not(mask)))

    # Now we repeat this procedure, but lower the detection limit to sigmalimlow :
    cosmics = growconvolve(cosmics)


    cosmics = np.logical_and(sp > sigcliplow, np.logical_and(cosmics, np.logical_not(mask)))
    # Our CR counter
    cdef numcr = cosmics.sum()


    # We update the crmask with the cosmics we have found :
    crmask[:, :] = np.logical_or(crmask, cosmics)[:, :]
    # We return the number of cr pixels
    # (used by function lacosmic)

    return numcr;
