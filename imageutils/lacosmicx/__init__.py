# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Optimized implementation of the LA Cosmic algorithm

Name : Lacosmicx
Author : Curtis McCully
Date : October 2014

 About
 =====

 Lacosmicx is designed to detect cosmic rays in images (numpy arrays),
 based on Pieter van Dokkum's L.A.Cosmic algorithm.

 Much of this was originally adapted from cosmics.py written by Malte Tewes.
 I have ported all of the slow functions to Cython/C, and optimized
 where I can. This is designed to be as fast as possible so some of the
 readability has been sacrificed, specifically in the C code.

 L.A.Cosmic = LAplacian Cosmic ray detection

 U{http://www.astro.yale.edu/dokkum/lacosmic/}

 (article : U{http://arxiv.org/abs/astro-ph/0108003})

 This code requires Cython, preferably version >= 0.21.

 Parallelization is achieved using OpenMP. This code should compile (although
 the Cython files may have issues) using a compiler that does not support OMP,
 e.g. clang.

 Differences from original LACosmic
 ===================

 - Automatic recognition of saturated stars, including their trails.
 This avoids treating such stars as large cosmic rays.

 -I have tried to optimize all of the code as much as possible while
 maintaining the integrity of the algorithm. One of the key speedups is to
 use a separable median filter instead of the true median filter. While these
 are not identical, they produce comparable results and the separable version
 is much faster.

 -This implementation is much faster than the Python by as much as a factor of
 17 depending on the given parameters, even without running multiple threads.
 With multiple threads, this can be increased easily by another factor of 2.
 This implementation is much faster than the original IRAF version
 (orders of magnitude).

 -The arrays always must be C-contiguous, thus all loops are y outer, x inner.
 Note that this follows the Pyfits convention.
 """
# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .lacosmicx import run
#    from ..laxwrappers import *
__all__ = ['run']
