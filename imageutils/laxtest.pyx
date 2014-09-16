# distutils: language = c++
# distutils: sources = functions.cpp

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
np.import_array()
from scipy import ndimage

from libcpp cimport bool

cdef extern from "functions.h":
    float median(float* a, int n)
    float optmed25(float* a)
    float optmed3(float* a)
    float optmed7(float* a)
    float optmed5(float* a)
    float optmed9(float* a)
    float* medfilt3(float* data, int nx, int ny)
    float* medfilt5(float* data, int nx, int ny)
    float* medfilt7(float* data, int nx, int ny)
    float* sepmedfilt3(float* data, int nx, int ny)
    float* sepmedfilt5(float* data, int nx, int ny)
    float* sepmedfilt7(float* data, int nx, int ny)
    float* sepmedfilt9(float* data, int nx, int ny)
    bool* dilate(bool* data, int iter, int nx, int ny)
    float* subsample(float* data, int nx, int ny)
    float* laplaceconvolve(float* data, int nx, int ny)
    float* rebin(float* data, int nx, int ny)
    bool* growconvolve(bool* data, int nx, int ny)
    void updatemask(float* data, bool* mask, float satlevel, int nx, int ny, bool fullmedian=false )
    int lacosmiciteration(float* cleanarr, bool* mask, bool* crmask, float sigclip, float objlim, float sigfrac, float backgroundlevel, float readnoise, int nx, int ny, bool fullmedian=false)

def test(d):
    return ctest(d, d.shape[0], d.shape[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ctest(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, nrows, ncols):
    cdef float* arrptr = <np.float32_t*> np.PyArray_DATA(d)
    cdef float* m3 = medfilt3(arrptr, nrows, ncols)
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:nrows, :ncols]> m3
    return np.asarray(m3_memview)


def pymedfilt3(d):
    return cymedfilt3(d,d.shape[1],d.shape[0])
cdef cymedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:ny, :nx]> medfilt3(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m3_memview)


def pymedfilt5(d):
    return cymedfilt5(d,d.shape[1],d.shape[0])
cdef cymedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m5_memview = <np.float32_t[:ny, :nx]> medfilt5(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m5_memview)

def pymedfilt7(d):
    return cymedfilt7(d,d.shape[1],d.shape[0])
cdef cymedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m7_memview = <np.float32_t[:ny, :nx]> medfilt7(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m7_memview)

def pysepmedfilt3(d):
    return cysepmedfilt3(d,d.shape[1],d.shape[0])
cdef cysepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m3_memview = <np.float32_t[:ny, :nx]> sepmedfilt3(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m3_memview)

def pysepmedfilt5(d):
    return cysepmedfilt5(d,d.shape[1],d.shape[0])
cdef cysepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m5_memview = <np.float32_t[:ny, :nx]> sepmedfilt5(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m5_memview)

def pysepmedfilt7(d):
    return cysepmedfilt7(d,d.shape[1],d.shape[0])
cdef cysepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m7_memview = <np.float32_t[:ny, :nx]> sepmedfilt7(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m7_memview)

def pysepmedfilt9(d):
    return cysepmedfilt9(d,d.shape[1],d.shape[0])
cdef cysepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] m9_memview = <np.float32_t[:ny, :nx]> sepmedfilt9(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(m9_memview)

def pydilate(d, niter):
    return cydilate(d,niter,d.shape[1],d.shape[0] )
def cydilate(np.ndarray[np.uint8_t, ndim=2, mode='c',cast=True] d, int niter, int nx, int ny):
    dilateptr =   dilate(<bool*> np.PyArray_DATA(d), niter, nx, ny )
    cdef bool[:,::1] dilatearr = np.zeros((ny,nx), dtype = np.uint8)
    for i in range(ny):
        for j in range(nx):
            dilatearr[i,j] = dilateptr[i*nx + j]
    #cdef bool[:, ::1] dilate_memview = <np.uint8_t[:ny, :nx]>
    return np.asarray(dilatearr,dtype=np.bool)

def pygrowconvolve(d):
    return cygrowconvolve(d,d.shape[1],d.shape[0] )
def cygrowconvolve(np.ndarray[np.uint8_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    growptr =   growconvolve(<bool*> np.PyArray_DATA(d), nx, ny )
    cdef bool[:,::1] growarr = np.zeros((ny,nx), dtype = np.uint8)
    for i in range(ny):
        for j in range(nx):
            growarr[i,j] = growptr[i*nx + j]
    #cdef bool[:, ::1] dilate_memview = <np.uint8_t[:ny, :nx]>
    return np.asarray(growarr,dtype=np.bool)

def pysubsample(d):
    return cysubsample(d,d.shape[1],d.shape[0])
cdef cysubsample(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] subsamp_memview = <np.float32_t[:2*ny, :2*nx]> subsample(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(subsamp_memview)

def pylaplaceconvolve(d):
    return cylaplaceconvolve(d,d.shape[1],d.shape[0])
cdef cylaplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] lap_memview = <np.float32_t[:ny, :nx]> laplaceconvolve(<float*> np.PyArray_DATA(d), nx, ny )
    return np.asarray(lap_memview)

def pyrebin(d):
    return cyrebin(d,d.shape[1],d.shape[0])
cdef cyrebin(np.ndarray[np.float32_t, ndim=2, mode='c',cast=True] d, int nx, int ny):
    cdef np.float32_t [:, ::1] rebin_memview = <np.float32_t[:ny/2, :nx/2]> rebin(<float*> np.PyArray_DATA(d),  nx/2, ny/2 )
    return np.asarray(rebin_memview)

import lacosmicx
import unittest
class Test(unittest.TestCase):
    "Function Tests"
    def test_median(self):
        a = np.ascontiguousarray(np.random.random(1001)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(median(<float*> np.PyArray_DATA(a),1001)))
    def test_optmed25(self):
        a = np.ascontiguousarray(np.random.random(25)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed25(<float*> np.PyArray_DATA(a))))
    def test_optmed3(self):
        a = np.ascontiguousarray(np.random.random(3)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed3(<float*> np.PyArray_DATA(a))))
    def test_optmed5(self):
        a = np.ascontiguousarray(np.random.random(5)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed5(<float*> np.PyArray_DATA(a))))
    def test_optmed7(self):
        a = np.ascontiguousarray(np.random.random(7)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed7(<float*> np.PyArray_DATA(a))))
    def test_optmed9(self):
        a = np.ascontiguousarray(np.random.random(9)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed9(<float*> np.PyArray_DATA(a))))
    def test_medfilt3(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed3 = ndimage.filters.median_filter(a,size=(3,3),mode='nearest')
        npmed3[:1,:]= a[:1,:]
        npmed3[-1:,:]= a[-1:,:]
        npmed3[:,:1]= a[:,:1]
        npmed3[:,-1:]= a[:,-1:]
        
        med3 = pymedfilt3(a) 
        self.assertEqual(1001*1001, (med3 == npmed3).sum())    
    
    def test_medfilt5(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed5 = ndimage.filters.median_filter(a,size=(5,5),mode='nearest')
        npmed5[:2,:]= a[:2,:]
        npmed5[-2:,:]= a[-2:,:]
        npmed5[:,:2]= a[:,:2]
        npmed5[:,-2:]= a[:,-2:]
        
        med5 = pymedfilt5(a) 
        self.assertEqual(1001*1001, (med5 == npmed5).sum())

    def test_medfilt7(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed7 = ndimage.filters.median_filter(a,size=(7,7),mode='nearest')
        npmed7[:3,:]= a[:3,:]
        npmed7[-3:,:]= a[-3:,:]
        npmed7[:,:3]= a[:,:3]
        npmed7[:,-3:]= a[:,-3:]
        
        med7 = pymedfilt7(a)
        self.assertEqual(1001*1001, (med7 == npmed7).sum())
    
    def test_sepmedfilt3(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed3 = ndimage.filters.median_filter(a,size=(1,3),mode='nearest')
        npmed3[:,:1]= a[:,:1]
        npmed3[:,-1:]= a[:,-1:]
        npmed3 = ndimage.filters.median_filter(npmed3,size=(3,1),mode='nearest')
        npmed3[:1,:]= a[:1,:]
        npmed3[-1:,:]= a[-1:,:]
        npmed3[:,:1]= a[:,:1]
        npmed3[:,-1:]= a[:,-1:]
        
        med3 = pysepmedfilt3(a) 
        self.assertEqual(1001*1001, (med3 == npmed3).sum())    
    
    def test_sepmedfilt5(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed5 = ndimage.filters.median_filter(a,size=(1,5),mode='nearest')
        npmed5[:,:2]= a[:,:2]
        npmed5[:,-2:]= a[:,-2:]
        npmed5 = ndimage.filters.median_filter(npmed5,size=(5,1),mode='nearest')
        npmed5[:2,:]= a[:2,:]
        npmed5[-2:,:]= a[-2:,:]
        npmed5[:,:2]= a[:,:2]
        npmed5[:,-2:]= a[:,-2:]
        
        med5 = pysepmedfilt5(a) 
        self.assertEqual(1001*1001, (med5 == npmed5).sum())    
        
    def test_sepmedfilt7(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed7 = ndimage.filters.median_filter(a,size=(1,7),mode='nearest')
        npmed7[:,:3]= a[:,:3]
        npmed7[:,-3:]= a[:,-3:]
        npmed7 = ndimage.filters.median_filter(npmed7,size=(7,1),mode='nearest')
        npmed7[:3,:]= a[:3,:]
        npmed7[-3:,:]= a[-3:,:]
        npmed7[:,:3]= a[:,:3]
        npmed7[:,-3:]= a[:,-3:]
        
        med7 = pysepmedfilt7(a) 
        self.assertEqual(1001*1001, (med7 == npmed7).sum())    

    def test_sepmedfilt9(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed9 = ndimage.filters.median_filter(a,size=(1,9),mode='nearest')
        npmed9[:,:4]= a[:,:4]
        npmed9[:,-4:]= a[:,-4:]
        npmed9 = ndimage.filters.median_filter(npmed9,size=(9,1),mode='nearest')
        npmed9[:4,:]= a[:4,:]
        npmed9[-4:,:]= a[-4:,:]
        npmed9[:,:4]= a[:,:4]
        npmed9[:,-4:]= a[:,-4:]
        
        med9 = pysepmedfilt9(a) 
        self.assertEqual(1001*1001, (med9 == npmed9).sum())    
    
    def test_dilate(self):
        #Put 5% of the pixels into a mask
        a = np.zeros((1001,1001),dtype=np.bool)
        a[np.random.random((1001, 1001)) <0.05] = True
        kernel = np.ones((5,5))
        kernel[0,0] = 0
        kernel[0,4] = 0
        kernel[4,0] = 0
        kernel[4,4] = 0
        #Make a zero padded array for the numpy version to operate
        paddeda = np.zeros((1005,1005),dtype=np.bool)
        paddeda[2:-2,2:-2]= a[:,:]
        npdilate = ndimage.morphology.binary_dilation(np.ascontiguousarray(paddeda),structure=kernel, iterations =2)
        cdilate = pydilate(a,2)
        
        self.assertEqual(1001*1001, (npdilate[2:-2,2:-2] == cdilate).sum())
        
    def test_growconvolve(self):
        #Put 5% of the pixels into a mask
        a = np.zeros((1001,1001),dtype=np.bool)
        a[np.random.random((1001, 1001)) <0.05] = True
        kernel = np.ones((3,3))
        npgrow = ndimage.morphology.binary_dilation(np.ascontiguousarray(a), structure = kernel,iterations =1)
        cygrow = pygrowconvolve(a)
        npgrow[:,0] = a[:,0]
        npgrow[:,-1] = a[:,-1]
        npgrow[0,:] = a[0,:]
        npgrow[-1,: ] = a[-1,:]
        self.assertEqual(1001*1001, (npgrow == cygrow).sum())
        
    def test_subsample(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npsubsamp = np.zeros((a.shape[0]*2, a.shape[1]*2), dtype=np.float32)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):    
                npsubsamp[2*i,2*j] = a[i,j]
                npsubsamp[2*i+1,2*j] = a[i,j]
                npsubsamp[2*i,2*j+1] = a[i,j]
                npsubsamp[2*i+1,2*j+1] = a[i,j]
        
        cysubsamp = pysubsample(a)
        self.assertEqual(2*1001*2*1001, (npsubsamp == cysubsamp).sum())

    def test_rebin(self):
        a = np.ascontiguousarray(np.random.random((2002, 2002))).astype('<f4')
        nprebin = np.zeros((1001,1001),dtype = np.float32)
        for i in range(1001):
            for j in range(1001):    
                nprebin[i,j] = (a[2*i,2*j] + a[2*i+1,2*j] + a[2*i,2*j+1] + a[2*i+1,2*j+1] )/  np.array([4.0],dtype=np.float32).astype('<f4')
                
        cyrebin = pyrebin(a)
        self.assertEqual(1001*1001, ((nprebin - cyrebin) < 1e-6).sum())
     
        
    def test_laplaceconvolve(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        k = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]]).astype('<f4')
        npconv = ndimage.filters.convolve(a, k, mode = 'constant', cval = 0.0 )
        cpyconv = pylaplaceconvolve(a)
        self.assertEqual(1001*1001, (npconv == cpyconv).sum())
        
        
    def runTest(self):
        self.test_median()
        self.test_optmed25()
        self.test_optmed3()
        self.test_optmed5()
        self.test_optmed7()
        self.test_optmed9()
        self.test_medfilt3()
        self.test_medfilt5()
        self.test_medfilt7()
        self.test_sepmedfilt3()
        self.test_sepmedfilt5()
        self.test_sepmedfilt7()
        self.test_dilate()
        self.test_growconvolve()
        self.test_subsample()
        self.test_rebin()
        
if __name__ == '__main__':
    unittest.main()