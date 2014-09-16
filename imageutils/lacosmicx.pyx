# distutils: language = c++
# distutils: sources = functions.cpp
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

cdef extern from "functions.h":
    float median(float* a, int n)
    float* medfilt3(float* arr, int nx, int ny)
    void updatemask(float* data, bool* mask, float satlevel, int nx, int ny, bool fullmedian)
    int lacosmiciteration(float* cleanarr, bool* mask, bool* crmask, float sigclip, float objlim, float sigfrac, float backgroundlevel, float readnoise, int nx, int ny, bool fullmedian)

def test(d):
    return ctest(d, d.shape[0], d.shape[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ctest(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, nrows, ncols):
    cdef float* arrptr = <np.float32_t*> np.PyArray_DATA(d)
    cdef float* m3 = medfilt3(arrptr, nrows, ncols)
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:nrows, :ncols]> m3
    return np.asarray(m3_memview)


#A python wrapper for the cython function for lacosmicx
def run(data, mask=None, sigclip = 4.5, sigfrac = 0.3, objlim = 5.0, readnoise = 6.5 , satlevel = 65536.0,
        pssl = 0.0, gain = 1.0, niter = 4, fullmedian = False):
    return np.asarray(crun(data, mask, sigclip=sigclip,sigfrac=sigfrac,objlim=objlim, readnoise = readnoise,
                           satlevel = satlevel, pssl = pssl, gain=gain, niter = niter, fullmedian=fullmedian), dtype=np.bool)

cdef bool[:,::1] crun(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] indat,
                      np.ndarray[np.uint8_t, ndim=2, mode='c',cast=True] inmask,
                       float sigclip = 4.5, float sigfrac  = 0.3, 
                       float objlim = 5.0, float readnoise =  6.5, float satlevel = 65536.0, 
                       float pssl =0.0, float gain=1.0 , 
                       int niter=4, fullmedian=False):
    cdef int nx = indat.shape[1]
    cdef int ny = indat.shape[0]
    #First it isn't good form to update the input data directly, so we make a copy
    cdef float[:,::1] data = np.empty_like(indat)
    for i in range(ny):
        for j in range(nx):
            data[i,j] = indat[i,j]
            
    #The data mask needs to be indexed with (i,j) -> (nx *j+i) (c-style indexed)
    cdef bool[:,::1] mask

    if inmask is None:
        #By default don't mask anything
        mask = np.zeros((ny,nx),dtype=np.uint8,order='C')
    else:
        #Make a copy of the input mask
        mask = np.empty_like(inmask,dtype=np.uint8)
        for i in range(ny):
            for j in range(nx):
                data[i,j] = indat[i,j]
                
    #Find the saturated stars and add them to the mask

    updatemask(<float*> np.PyArray_DATA(np.asarray(data)),<bool*> np.PyArray_DATA(np.asarray(mask)),satlevel, nx,ny, fullmedian)
    
    gooddata = np.zeros(nx*ny-np.asarray(mask).sum(),dtype=np.float32,order='C')

    igoodpix = 0

    #note the c-style convention here with y as the slow direction and x and the fast direction
    #This follows the pyfits convention as well

    for i in range(ny):
        for j in range(nx):
            # internally, we will always work "with sky" and in electrons, not ADU (gain=1)
            data[i,j] += pssl;
            data[i,j] *= gain;
            #Get only data that is unmasked
            if not mask[i,j]:
                gooddata[igoodpix] = data[i,j]
                igoodpix+=1
    #Get the default background level for large cosmics
    backgroundlevel = median( <float*> np.PyArray_DATA(np.asarray(gooddata)), len(gooddata))
    del gooddata

    #Defined a cleaned array
    #This is what we work with in lacosmic iteration
    #Set the initial values to those of the data array
    cdef float[:,::1] cleanarr = np.empty_like(indat)
    for i in range(ny):
        for j in range(nx):
            cleanarr[i,j]=data[i,j]

    #Define a cosmic ray mask
    #This is what will be returned at the end
    cdef bool[:,::1] crmask = np.zeros((ny,nx),dtype=np.uint8,order='C')

    #Run lacosmic for up to maxiter iterations
    #We stop if no more cosmics are found

    print "Starting {} L.A.Cosmic iterations".format(niter)
    for i in range(niter):
        print "Iteration {}:".format(i+1)

        #Detect the cosmic rays
        ncrpix = lacosmiciteration(<float*> np.PyArray_DATA(np.asarray(cleanarr)), <bool*> np.PyArray_DATA(np.asarray(mask)), <bool*> np.PyArray_DATA(np.asarray(crmask)),
                                    sigclip, objlim, sigfrac, backgroundlevel, readnoise,  nx,  ny,  fullmedian) 

        print "{} cosmic pixels this iteration".format(ncrpix)

        #If we didn't find anything, we're done.
        if ncrpix == 0:
            break

    return crmask
