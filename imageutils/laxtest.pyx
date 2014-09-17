
from laxwrappers import *
import numpy as np
cimport numpy as np
cimport cython
import unittest
from cython cimport floating
np.import_array()
from scipy import ndimage

from libcpp cimport bool

class Test(unittest.TestCase):
    "Function Tests"
    def test_median(self):
        a = np.ascontiguousarray(np.random.random(1001)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(median(a)))
        
    def test_optmed25(self):
        a = np.ascontiguousarray(np.random.random(25)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed25(a)))
    
    def test_optmed3(self):
        a = np.ascontiguousarray(np.random.random(3)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed3(a)))
    
    def test_optmed5(self):
        a = np.ascontiguousarray(np.random.random(5)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed5(a)))
    
    def test_optmed7(self):
        a = np.ascontiguousarray(np.random.random(7)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed7(a)))
    
    def test_optmed9(self):
        a = np.ascontiguousarray(np.random.random(9)).astype('<f4')
        self.assertEqual(np.float32(np.median(a)),np.float32(optmed9(a)))
        
    def test_medfilt3(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed3 = ndimage.filters.median_filter(a,size=(3,3),mode='nearest')
        npmed3[:1,:]= a[:1,:]
        npmed3[-1:,:]= a[-1:,:]
        npmed3[:,:1]= a[:,:1]
        npmed3[:,-1:]= a[:,-1:]
        
        med3 = medfilt3(a) 
        self.assertEqual(1001*1001, (med3 == npmed3).sum())    
    
    def test_medfilt5(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed5 = ndimage.filters.median_filter(a,size=(5,5),mode='nearest')
        npmed5[:2,:]= a[:2,:]
        npmed5[-2:,:]= a[-2:,:]
        npmed5[:,:2]= a[:,:2]
        npmed5[:,-2:]= a[:,-2:]
        
        med5 = medfilt5(a) 
        self.assertEqual(1001*1001, (med5 == npmed5).sum())

    def test_medfilt7(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        npmed7 = ndimage.filters.median_filter(a,size=(7,7),mode='nearest')
        npmed7[:3,:]= a[:3,:]
        npmed7[-3:,:]= a[-3:,:]
        npmed7[:,:3]= a[:,:3]
        npmed7[:,-3:]= a[:,-3:]
        
        med7 = medfilt7(a)
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
        
        med3 = sepmedfilt3(a) 
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
        
        med5 = sepmedfilt5(a) 
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
        
        med7 = sepmedfilt7(a) 
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
        
        med9 = sepmedfilt9(a) 
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
        cdilate = dilate(a,2)
        
        self.assertEqual(1001*1001, (npdilate[2:-2,2:-2] == cdilate).sum())
        
    def test_growconvolve(self):
        #Put 5% of the pixels into a mask
        a = np.zeros((1001,1001),dtype=np.bool)
        a[np.random.random((1001, 1001)) <0.05] = True
        kernel = np.ones((3,3))
        npgrow = ndimage.morphology.binary_dilation(np.ascontiguousarray(a), structure = kernel,iterations =1)
        cygrow = growconvolve(a)
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
        
        cysubsamp = subsample(a)
        self.assertEqual(2*1001*2*1001, (npsubsamp == cysubsamp).sum())

    @cython.cdivision(True)
    def test_rebin(self):
        a = np.ascontiguousarray(np.random.random((2002, 2002)),dtype=np.float32).astype('<f4')
        cdef float[:,::1] nprebin = np.zeros((1001,1001),dtype = np.float32).astype('<f4')
        for i in range(1001):
            for j in range(1001):    
                nprebin[i,j] = (a[2*i,2*j] + a[2*i+1,2*j] + a[2*i,2*j+1] + a[2*i+1,2*j+1])/np.float32(4.0)
                
        cyrebin = rebin(a)
        #self.assertEqual(1001*1001, (nprebin == cyrebin).sum())
        #self.assertEqual(1001*1001, (abs(np.asarray(nprebin) -cyrebin) < 1e-6).sum())
     
        
    def test_laplaceconvolve(self):
        a = np.ascontiguousarray(np.random.random((1001, 1001))).astype('<f4')
        k = np.array([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]]).astype('<f4')
        npconv = ndimage.filters.convolve(a, k, mode = 'constant', cval = 0.0 )
        cyconv = laplaceconvolve(a)
        self.assertEqual(1001*1001, (npconv == cyconv).sum())
        
        
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