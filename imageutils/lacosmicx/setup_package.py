from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_: ['coveragerc']}


def get_extensions():
        cyExts = cythonize("imageutils/lacosmicx/la*.pyx")
        for ext in cyExts:
            ext.include_dirs = [np.get_include()]
            ext.extra_compile_args = ['-g', '-O3', '-fopenmp',
                                      '-funroll-loops', '-ffast-math']
            ext.libraries = ['gomp']
            ext.extra_link_args = ['-g', '-fopenmp']
            ext.sources.append('imageutils/lacosmicx/laxutils.c')
#            ext.define_macros = [("NPY_NO_DEPRECATED_API",
#                                  "NPY_1_7_API_VERSION")]
            print(ext.sources)

        return cyExts
