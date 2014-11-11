import os

from distutils.core import setup, Extension
from distutils.ccompiler import new_compiler

# check to see if openmp is supported
USE_OPENMP = False
ccompiler = new_compiler()
ccompiler.add_library('gomp')
try:
   USE_OPENMP=ccompiler.has_function('omp_get_num_threads')
except:
   USE_OPENMP=False


LACOSMICX_ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    sources = [os.path.join(LACOSMICX_ROOT, "lacosmicx.pyx"),
               os.path.join(LACOSMICX_ROOT, "laxutils.c")]

    include_dirs = ['numpy', LACOSMICX_ROOT]

    libraries = []

    ext = Extension(name="imageutils.lacosmicx.lacosmicx",
                    sources=sources,
                    include_dirs=include_dirs,
                    libraries=libraries,
                    language="c",
                    extra_compile_args=['-g', '-O3',
                                        '-funroll-loops', '-ffast-math'])

    if USE_OPENMP:
        ext.extra_compile_args.append('-fopenmp')
        ext.extra_link_args = ['-g', '-fopenmp']

    return [ext]
