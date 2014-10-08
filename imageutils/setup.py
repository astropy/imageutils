from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

cyExts  = cythonize("la*.pyx")
for ext in cyExts:
    ext.include_dirs = [np.get_include()]
    ext.extra_compile_args =['-O3','-fopenmp','-funroll-loops','-ffast-math']   
    ext.libraries = ['gomp']
    ext.extra_link_args=['-fopenmp']
setup(ext_modules=cyExts)
exit()
