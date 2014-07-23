**********
imageutils
**********

Image processing utilities for Astropy.

* Code: https://github.com/astropy/imageutils
* Docs: https://imageutils.readthedocs.org/

What is this?
-------------

This is an attempt to collect image processing utilities that are generally considered
useful for astronomers and propose to include them as ``astropy.image`` into the
Astropy core package in fall 2014 (before the Astropy 1.0 release).

The need for this became clear when Astropy affiliated packages started to re-implement
common image processing functions like resampling or cutouts over and over.
Note that there is a separate repo for `reproject` functionality
and there's the `astropy.convolution` sub-package that's already in the core.

The philosophy here should be to use `numpy`,
`scipy.ndimage` and `skimage` as much as possible instead of re-implementing basic
array and image utility functions. In many cases we will write a wrapper here that
calls the corresponding function e.g. in `skimage` or `scipy.ndimage`, but in addition has
an `astropy.wcs.WCS` object as input and output and updates it accordingly (think e.g. downsample or cutout). 

Contributions welcome!
(please start by filing an `issue on Github <https://github.com/astropy/imageutils/issues>`__ asking if
some functionality is in scope for this package before spending time on a pull request)


.. automodapi:: imageutils
