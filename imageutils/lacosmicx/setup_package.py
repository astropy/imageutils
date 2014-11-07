import os

from distutils.core import setup, Extension


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
                    extra_compile_args=['-g', '-O3', '-fopenmp',
                                        '-funroll-loops', '-ffast-math'])

    # TODO: figure out how to put these only when not using clang:
    # ext.extra_link_args = ['-g', '-fopenmp']

    return [ext]
