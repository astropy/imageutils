.. _api-spec:

******************************
`imageutils` API specification
******************************

`imageutils` contains common astronomical image utilities.

Affiliated package maintainers and users are encouraged to use this style to write
their own image utils function. 

The `imageutils` API mainly consists of functions, it doesn't define an extra ``Image`` class.

What's an image?
----------------

An image is either

1. a `numpy.ndarray`
2. a tuple consisting of a `numpy.ndarray` and an `astropy.wcs.WCS` object
3. an `astropy.nddata.NDData` object
   that can optionally contain `mask`, `flags`, `uncertainty` attributes
   as well as a `wcs` attribute.

.. note:: `astropy.nddata.NDData` will probably change significantly from Astropy 0.4 -> 1.0,
          so this description is very preliminary ... `imageutils` will adapt to any changes
          (and in some cases contribute to the dicussion how `NDData` should work).

Proposals
---------

At this point we have a few proposals on the table ... we should agree on one scheme soon
so that we can start to work on the details and implementation.


Here's some useful examples of existing functions to look at:

* `ccdproc.rebin <http://ccdproc.readthedocs.org/en/latest/_modules/ccdproc/core.html#rebin>`__
* `reproject.reproject <http://reproject.readthedocs.org/en/latest/_modules/reproject/high_level.html#reproject>`__
* please add a few more relevant examples here.

Proposal 1
..........

I think this is approximately what @keflavich and @cdeil had in mind after the discussion
in https://github.com/astropy/imageutils/issues/2 :

.. code-block:: python

   def downsample(image, wcs, factor):
      """Downsample image by a given factor.
      
      Parameters
      ----------
      array : `numpy.ndarray`
         Input array
      wcs : `astropy.wcs.WCS`
         Input wcs
      factor : int
         Downsample factor
      
      Returns
      -------
      out_array : `numpy.ndarray`
         Output array
      out_wcs : `astropy.wcs.WCS`
         Output wcs
      """
      # Compute `out_array`
      from skimage.measure import block_reduce
      out_array = block_reduce(array, factor)
      
      # Compute `out_wcs`
      out_wcs = wcs.copy()
      if hasattr(wcs_out.wcs, 'cd'):
         out_wcs.wcs.cdelt = out_wcs.wcs.cdelt * factor
      else:
         assert np.all(wcs_out.wcs.cdelt == 1.0)
         wcs_out.wcs.pc = wcs_out.wcs.pc * factor
         wcs_out.wcs.crpix =  ((wcs_out.wcs.crpix-0.5)/factor)+0.5

      return out_array, out_wcs 


Proposal 2
..........

When trying to support `astropy.nddata.NDData` things get more complicated very quickly
(because of all the stuff that is (optionally) in `NDData`, namely `mask`, `uncertainty`, `wcs`, flags`.

I don't know how to do this well ... please make proposals (preferably via pull requests against this one):

.. code-block:: python

   def downsample(image, factor):
      """Downsample image by a given factor.
      
      Parameters
      ----------
      image : `numpy.ndarray` or tuple (`numpy.ndarray`, `astropy.wcs.WCS`) or `astropy.nddata.NDData`
         Image
      factor : int
         Downsample factor
      
      Returns
      -------
      out_image : typeof(image)
         Image
      """
      # Always start by converting to the most general case: `NDData`
      # ... that's just one option to do it ... is this what we want???
      nddata = prepare_imageutils_input(image)

      # downsample `nddata`
      # The key is not to duplicate the code here for the different input types.
      # There should be another utility function to automatically process all "images" in the
      # `NDData` object, i.e. the `data`, `mask` and all the images in `flags`
      # and maybe `uncertainty` to avoid boilerplate code.
      nddata_out = ...
      
      # Return image of the same type that was put in.
      out_image = nddata_to_image_type(nddata, image)
      return out_image

Proposal 3
..........

Always convert to `astropy.nddata.NDData` at the start and then return an `astropy.nddata.NDData` object.
This effectively means we have agreed on an ``Image`` object ... images are represented by ``NDData`` objects. 

Notes
-----

* `imageutils` functions should never modify their inputs!
  I.e. they don't do in-place operations, but start by creating copies or new objects. 

To be discussed
---------------

Here's some of the questions that we should discuss ... please add

* Should we ever accept / return `astropy.io.fits.ImageHDU`, `astropy.io.fits.Header` or
  `astropy.io.fits.HDUList` objects as a convenience for the user?
* What's the API for functions with multiple input images or output images?
* Are we agreed that imageutils functions should not do error propagation, this is left up to the caller?
* Should / can we support `astropy.utils.Quantity` objects for all / most image utils functions?
* Should we support masked numpy arrays or require that user that need masks use `NDData` objects
  (which have a `mask` attribute)?