from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ : ['coveragerc']}

def get_extensions():
        cyExts  = cythonize("imageutils/lacosmicx/la*.pyx")
        for ext in cyExts:
            ext.include_dirs = [np.get_include()]
            ext.extra_compile_args =['-O3','-fopenmp','-funroll-loops','-ffast-math']   
            ext.libraries = ['gomp']
            ext.extra_link_args=['-fopenmp']
            ext.sources.append('imageutils/lacosmicx/laxutils.c')
            print(ext.sources)

        return cyExts
