import os

from distutils.core import setup, Extension


# TODO: figure out how to set this conditionally for compilers that support it
USE_OPENMP = False


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
