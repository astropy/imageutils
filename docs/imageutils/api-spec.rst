.. _api-spec:

******************************
`imageutils` API specification
******************************

`imageutils` (will be proposed for inclusion as ``astropy.image`` in the Astropy core)
contains common astronomical image utilities.

Affiliated package maintainers and users are encouraged to use this style to write
their own image utils function. 

The `imageutils` API mainly consists of functions, it doesn't define an extra ``Image`` class.

What's an image?
----------------

.. note:: This is not a definition ... it was just an explanation for which objects
          we are considering to use as inputs / outputs.

An image is either

1. a `numpy.ndarray`
2. a tuple consisting of a `numpy.ndarray` and an `astropy.wcs.WCS` object
3. an `astropy.nddata.NDData` object
   that can optionally contain ``mask``, ``flags``, ``uncertainty`` attributes
   as well as a ``wcs`` attribute.

.. note:: `astropy.nddata.NDData` will probably change significantly from Astropy 0.4 -> 1.0,
          so this description is very preliminary ... `imageutils` will adapt to any changes
          (and in some cases contribute to the dicussion how `NDData` should work).

Examples
--------

Here's some useful examples of existing image utility functions to look at
(note that they don't conform to a uniform API yet):

* `ccdproc.rebin <http://ccdproc.readthedocs.org/en/latest/_modules/ccdproc/core.html#rebin>`__
* `reproject.reproject <http://reproject.readthedocs.org/en/latest/_modules/reproject/high_level.html#reproject>`__
* please add a few more relevant examples here.

To have concrete examples in mind, these are some of the functions we want to implement in ``astropy.image``: 

* downsample / upsample / block-reduce / zoom (see issues #2, #5, #8)
* `crop = cutout (see issue #4) <https://github.com/astropy/imageutils/issues/4>`__
* `Image scaling (see issue #7) <https://github.com/astropy/imageutils/pull/17>`__
* `Helper functions to make empty images (e.g. to simulate for model data)
  (see issue #7) <https://github.com/astropy/imageutils/issues/7>`__
* `LACosmic cosmic ray removal (see issue #12) <https://github.com/astropy/imageutils/issues/12>`__

We could have more astronomy-oriented algorithms like SExtractor-style background estimation
(already implemented by @kbarbary in `sep-python <https://github.com/kbarbary/sep-python>`__).

Proposals
---------

At this point we have a few proposals on the table ... we should agree on one scheme soon
so that we can start to work on the details and implementation.



Proposal 1
..........

.. note::
   This is approximately what @keflavich and @cdeil had in mind after the discussion
   in https://github.com/astropy/imageutils/issues/2 .
   
   @ejeschke also proposed a two-level scheme in
   https://github.com/astropy/imageutils/pull/18#issuecomment-52163028
   but he suggested to put the processing of `uncertainty`, `mask`, `flags` into
   the low-level function (where here the proposal is to put it in the high-level
   function or an OO method if possible). 

The basic idea in this proposal is to have the basic implementation of each image utility
function have an `~numpy.ndarray` and a `~astropy.wcs.WCS` as input and output.

And to be as modular / reusable as possible by putting the code that computes the output
`~numpy.ndarray` and the `~astropy.wcs.WCS` object into separate functions. 

.. code-block:: python
   :linenos:

   def downsample(array, factor, wcs=None):
       """Downsample image by a given factor.
   
       Parameters
       ----------
       array : `numpy.ndarray`
          Input array
       factor : int
          Downsample factor
       wcs : `astropy.wcs.WCS`, optional
          Input wcs
   
       Returns
       -------
       out_array : `numpy.ndarray`
          Output array
       out_wcs : `astropy.wcs.WCS`
          Output wcs
       """
       array_out = _downsample_array(array, factor)
   
       if wcs:
           wcs_out = _downsample_wcs(wcs, factor)
           return array_out, wcs_out
       else:
           return array_out
   
   
   def _downsample_array(array, factor):
       # The image array methods will usually call or many
       # numpy, scipy.ndimage or scikit-image functions,
       # but could also implement the algorithm directly
       # in Python or Cython in `imageutils`
       from skimage.measure import block_reduce
       # This will make a copy and leave the input array unchanged
       array_out = block_reduce(array, factor)
       return array_out
   
   
   def _downsample_wcs(wcs, factor):
       wcs_out = wcs.copy()
   
       if hasattr(wcs_out.wcs, 'cd'):
           wcs_out.wcs.cdelt = wcs_out.wcs.cdelt * factor
       else:
           assert np.all(wcs_out.wcs.cdelt == 1.0)
           wcs_out.wcs.pc = wcs_out.wcs.pc * factor
           wcs_out.wcs.crpix = ((wcs_out.wcs.crpix - 0.5) / factor) + 0.5
   
       return wcs_out

Based on this one could then implement functions or method for `~astropy.nddata.NDData`,
although I think this will become problematic quickly for several reasons:

#. If `~astropy.nddata.NDData` gets a new property (say e.g. a systematic uncertainty array),
   all image utility functions need to be updated to handle it!
#. There might be different use cases / no agreement on the semantics of how to
   handle the ``uncertainty``, ``mask`` and ``flags``
   ... even for such a simple function as ``downsample`` people want to combine
   ``data``, ``uncertainty``, ``mask`` and ``flags`` in different ways
   ... maybe this is a hint to leave the implementation to the user (e.g. the ``CCDData`` class)?
#. We will have to write and maintain maybe 2 to 3 times as much code to handle all the
   possible NDData attributes (lots of ``isinstance``, ``hasattr``, ``if``, ``try``)
   ... maybe better to leave this to others (i.e. ``NDData``, ``CCDData``, ``SpectralCube``, ...).

Here's what an implementation could look like (incomplete):

.. code-block:: python
   :linenos:

   def downsample_nddata(nddata, factor):
       """Downsample NDData object.
   
       Parameters
       ----------
       nddata : `~astropy.nddata.NDData`
          Input nddata
       factor : int
          Downsample factor
   
       Returns
       -------
       out_nddata : `~astropy.nddata.NDData`
          Output nddata
       """
       # Compute all attributes of `out_nddata`
       # skipping optional things
       data = _downsample_array(nddata.data, factor)
   
       if hasattr(nddata, 'uncertainty'):
           uncertainty = _downsample_array(nddata.uncertainty, factor)
       else:
           uncertainty = None
   
       if hasattr(nddata, 'mask'):
           mask = _downsample_array(nddata.mask, factor)
           # TODO: what should the value of the output mask be
           # if some corresponding pixels in the input are masked?
           mask = np.where(mask > 0, 1, 0)
       else:
           mask = None
   
       if hasattr(nddata, 'wcs'):
           wcs = _downsample_wcs(nddata.wcs, factor)
       else:
           wcs = None
   
       nddata_out = NDData(data=data,
                           uncertainty=uncertainty,
                           mask=mask,
                           flags=flags,
                           wcs=wcs,
                           meta=nddata.meta.copy(),
                           unit=nddata.unit.copy(),
                           )

The nice thing about this separation is that we can implement the simple image utility function
**now** and the ``*_nddata`` versions can come later if we figure out how to implement them
so that they work for many / all use-cases and subclasses.

Proposal 2
..........

.. note:: This proposal was just a question and has been withdrawn.

Proposal 3
..........

.. note:: This proposal was just a question and has been withdrawn.

Proposal 4 (by Erik)
....................

.. note::
   This proposal was made by Erik in https://github.com/astropy/imageutils/pull/18#issuecomment-51725518 

@cdeil - I'm not a fan of the definition of image (which sort of carries down to the rest), because of the (data, wcs) tuple option. So I'd like to proposal an alternative, more in line with what @astrofrog and @ericjesche and I were suggesting in #13 (and I think also compatible with @mwcraig's request?) about duck-typing:

An image is either:

1. A numpy array
2. Something that has the same attributes as NDData.

Practically speaking, that leads to the following as implementation:

.. code-block:: python
   :linenos:

   def downsample(image, factor):
       """Downsample image by a given factor.
   
       Parameters
       ----------
       image
           ... description here ...
       factor : int
           Downsample factor
   
       Returns
       -------
       out : whatever `image` is
           Output
       """
       from copy import deepcopy
       from skimage.measure import block_reduce
   
       nddata_like_input = False
   
       #need to special-case ndarray, because ndarray *has* a `data` object, 
       # but it means the actual underlying memory
       if hasattr(image, 'data') and not isinstance(image, np.ndarray):  
           inarray = image.data
           nddata_like_input = True
       else:
           inarray = image
   
       # Compute `out_array`
       out_array = block_reduce(inarray, factor)
   
       out_wcs = None
       if hasattr(image, 'wcs'):
           # Compute `out_wcs`
           out_wcs = image.wcs.copy()
           if hasattr(wcs_out.wcs, 'cd'):
               out_wcs.wcs.cdelt = out_wcs.wcs.cdelt * factor
           else:
               assert np.all(wcs_out.wcs.cdelt == 1.0)
               wcs_out.wcs.pc = wcs_out.wcs.pc * factor
               wcs_out.wcs.crpix =  ((wcs_out.wcs.crpix-0.5)/factor)+0.5
   
       if nddata_like_input:
           if hasattr(input, 'copy'):
               out = input.copy()
           else:
               out = deepcopy(out)
   
           out.data = output_array
           if out_wcs is not None:
               out.wcs = out_wcs
           
           # TODO: here code needs to be added to handle
           # `mask`, `uncertainty` and `flags`, right?
   
       else:
           out = output_array
   
       return out

That is, if an image is passed in, you just do the basic thing. You use ``hasattr(input, 'whatever')``
to check for ``mask``, ``wcs``, ``flags``, etc., and treat them like ``NDData`` equivalents.
Then you generate an output that's just whatever the input was.

An alternative would be to replace the copying for the output above with some sort of heuristic like:

``out=input.__class__(data=output_array, wcs=out_wcs)``
but that means any additional information is lost (like ``meta``, for example)


Proposal 5 (by Larry)
.....................

.. note:: This proposal was made by Larry in
          https://github.com/astropy/imageutils/pull/18#issuecomment-51948108 .

I like @eteq's version.

For discussion, I include an alternate implementation below that uses a recursive function call
(it may not be better than @eteq's version and it probably can be improved).
Essentially it unpacks the NDData-like object, calls the function with the nddarray/wcs inputs
and then creates an NDData-like object as output (all done in one if block).
It has the advantage of not making copies of the inputs
(e.g. data image, and optional mask and uncertainty images).
In this example I've included optional mask and uncertainty inputs, even though they do nothing here.
Also, I've included a wcs optional parameter. To use a recursive call, obviously the function needs to
have a parameter for every NDData-like attribute that is used in the function.

.. code-block:: python
   :linenos:

   def downsample(image, factor, wcs=None, uncertainty=None, mask=None):
       """Downsample image by a given factor.
   
       Parameters
       ----------
       image
           ... description here ...
       factor : int
           Downsample factor
       wcs : optional
           ... description here ...
       uncertainty : optional
           ... description here ...
       mask : optional
           ... description here ...
   
       Returns
       -------
       out : whatever `image` is
           Output
       out_wcs : WCS
           Returned *only* if input `image` is a `ndarray`
       """
   
       from skimage.measure import block_reduce
   
       # the code in this "if" block handles everything with NDData-like input
       # need to special-case ndarray, because ndarray *has* a `data` object,
       # but it means the actual underlying memory
       if hasattr(image, 'data') and not isinstance(image, np.ndarray):
           attribs = ['wcs', 'uncertainty', 'mask']
           inputs = {}
           for attrib in attribs:
               if hasattr(image, attrib):
                   inputs[attrib] = getattr(image, attrib)
               else:
                   inputs[attrib] = None
           out_array, out_wcs = downsample(image.data, factor, wcs=inputs['wcs'],
                                           uncertainty=inputs['uncertainty'],
                                           mask=inputs['mask'])
   
           if hasattr(image, 'copy'):
               out = image.copy()
           else:
               out = deepcopy(image)
           out.data = out_array
           if out_wcs is not None:
               out.wcs = out_wcs
           return out
   
       # Compute `out_array`
       out_array = block_reduce(image, factor)
   
       out_wcs = None
       if wcs is not None:
           # Compute `out_wcs`
           out_wcs = wcs.copy()
           if hasattr(out_wcs, 'cd'):
               out_wcs.cdelt *= factor
           else:
               assert np.all(out_wcs.cdelt == 1.0)
               out_wcs.pc *= factor
               out_wcs.crpix = ((out_wcs.crpix-0.5) / factor) + 0.5
   
       return out_array, out_wcs

Notes
-----

* `imageutils` functions should never modify their inputs!
  I.e. they don't do in-place operations, but start by creating copies or new objects. 

To be discussed
---------------

#. If we support ``NDData`` - like input, what should the semantics be how to combine
   ``data``, ``uncertainty``, ``mask`` and ``flags`` e.g. for the ``downsample`` image
   utility function?
#. What if more arrays are added to ``NDData`` or some of it's sub-classes in the future?
   Then the ``imageutils`` functions don't work correctly any more, because they don't
   process those extra arrays.
   So we either have to declare that ``NDData`` can only grow extra arrays in ``flags``
   (which could be a weird name if the extra arrays don't represent flags),
   or we need a list or dict of arrays as an ``NDData`` property to make it extensible.  
#. Implement each image utils function like ``downsample`` via two smaller functions
   ``downsample_wcs`` and ``downsample_array``?
   Advantage: Smaller functions means smaller complexity and more modularity?
#. Provide ``downsample_nddata(nddata)`` as a separate function that calls the simple``downsample(array, wcs)``
   (repeatedly for it's ``data``, ``mask``, ``uncertainty``, ``flags``, ... whatever we add in the future).
   Advantage: Smaller functions, more modular. Probably the ``*_nddata`` functions could largely be boilerplate
   code that loops over it's arrays, unless arrays need to be combined in complex ways
   (e.g. ``data`` and ``uncertainty``).
#. Should we ever accept / return `astropy.io.fits.ImageHDU`, `astropy.io.fits.Header` or
   `astropy.io.fits.HDUList` objects as a convenience for the user?
   Or should we "only" use documentation to tell users how to process FITS data with ``imageutils``
   (see https://github.com/astropy/imageutils/issues/3)?
#. Should / can we support `astropy.utils.Quantity` objects for all / most image utils functions?
#. Should we support masked numpy arrays or require that user that need masks use `NDData` objects
   (which have a `mask` attribute) or convert to masked numpy arrays `NDData` objects on input?
