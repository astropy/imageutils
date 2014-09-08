from distutils.core import setup, Extension
from Cython.Build import cythonize

#setup(ext_modules=cythonize("lacosmicx.pyx"))
cyExts  = cythonize("test.pyx", sources=["functions.cpp"],language="c++")
cyExts[0].include_dirs = ['/usr/local/astro64/include', '/usr/include/malloc',
                          '/usr/local/astro64/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include']
cyExts[0].extra_compile_args =['-O3','-fopenmp','-funroll-loops','-ffast-math']   
#cyExts[0].extra_compile_args =['-O3','-fopenmp']

cyExts[0].libraries = ['gomp','cfitsio']
cyExts[0].library_dirs = ['/usr/local/astro64/lib']
cyExts[0].extra_link_args=['-fopenmp']
setup(ext_modules=cyExts)
exit()
module1 = Extension('_lacosmicx',                    
                    include_dirs = ['/usr/local/astro64/include','/usr/stsci/pyssgx/Python2.7.1/include/python2.7','/usr/stsci/pyssgx/2.7.1/numpy/core/include/numpy','/usr/include/malloc'],
                    libraries = ['cfitsio','gomp'],
                    library_dirs = ['/usr/local/astro64/lib'],
                    extra_compile_args=['-O3','-fopenmp','-funroll-loops','-ffast-math','-sse','sse2'],
                    sources = ['functions.cpp']
                    )

setup (name = 'Lacosmicx',
       version = '1.0',
       author = 'Curtis McCully',
       author_email = 'cmccully@physics.rutgers.edu',
       ext_modules = module2)
