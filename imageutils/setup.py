from distutils.core import setup, Extension
from Cython.Build import cythonize

#setup(ext_modules=cythonize("lacosmicx.pyx"))
cyExts  = cythonize("la*.pyx")
for ext in cyExts:
    ext.include_dirs = ['/usr/local/astro64/include', '/usr/include/malloc',
                          '/usr/local/astro64/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include']
    ext.extra_compile_args =['-O3','-fopenmp','-funroll-loops','-ffast-math']   
    ext.libraries = ['gomp']
    ext.library_dirs = ['/usr/local/astro64/lib']
    ext.extra_link_args=['-fopenmp']
setup(ext_modules=cyExts)
exit()
