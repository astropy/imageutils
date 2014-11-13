from __future__ import division, print_function


from astropy.extern import six
from astropy.utils.misc import InheritDocstrings

__all__ = ["BaseTransform", "CompositeTransform"]

class BaseTransform(object):
    def __add__(self, other):
        return CompositeTransform(other, self)


class CompositeTransform(BaseTransform):
    """
    A combination of two transformes.

    Parameters
    ----------
    transform_1:
        The first transform to apply.
    transform_2:
        The second transform to apply.
    """

    def __init__(self, transform_1, transform_2):
        super(CompositeTransform, self).__init__()
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __call__(self, values, clip=True):
        return self.transform_2(self.transform_1(values, clip=clip), clip=clip)

    def inverted(self):
        return CompositeTransform(self.transform_2.inverted(),
                                  self.transform_1.inverted())
