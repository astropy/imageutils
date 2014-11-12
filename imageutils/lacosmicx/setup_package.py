import os
import sys
import subprocess

from distutils.core import Extension

LACOSMICX_ROOT = os.path.relpath(os.path.dirname(__file__))

CODELINES = """
import sys
from distutils.ccompiler import new_compiler
ccompiler = new_compiler()
ccompiler.add_library('gomp')
sys.exit(int(ccompiler.has_function('omp_get_num_threads')))
"""


def has_openmp():
    s = subprocess.Popen([sys.executable], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    s.communicate(CODELINES.encode('utf-8'))
    s.wait()
    return bool(s.returncode)


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

    if has_openmp():
        ext.extra_compile_args.append('-fopenmp')
        ext.extra_link_args = ['-g', '-fopenmp']

    return [ext]
