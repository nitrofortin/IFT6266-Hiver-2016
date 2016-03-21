"""Objects for encapsulating parameter initialization strategies."""
from abc import ABCMeta, abstractmethod
import numbers

import numpy
import theano
from six import add_metaclass


@add_metaclass(ABCMeta)
class NdarrayInitialization(object):
    """Base class specifying the interface for ndarray initialization."""
    @abstractmethod
    def generate(self, rng, shape):
        """Generate an initial set of parameters from a given distribution.

        Parameters
        ----------
        rng : :class:`numpy.random.RandomState`
        shape : tuple
            A shape tuple for the requested parameter array shape.

        Returns
        -------
        output : :class:`~numpy.ndarray`
            An ndarray with values drawn from the distribution specified by
            this object, of shape `shape`, with dtype
            :attr:`config.floatX`.

        """

    def initialize(self, var, rng, shape=None):
        """Initialize a shared variable with generated parameters.

        Parameters
        ----------
        var : object
            A Theano shared variable whose value will be set with values
            drawn from this :class:`NdarrayInitialization` instance.
        rng : :class:`numpy.random.RandomState`
        shape : tuple
            A shape tuple for the requested parameter array shape.

        """
        if not shape:
            shape = var.get_value(borrow=True, return_internal_type=True).shape
        var.set_value(self.generate(rng, shape))

        
class IsotropicGaussian(NdarrayInitialization):
    """Initialize parameters from an isotropic Gaussian distribution.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0

    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!

    """
    def __init__(self, std=1, mean=0):
        self._mean = mean
        self._std = std

    def generate(self, rng, shape):
        m = rng.normal(self._mean, self._std, size=shape)
        return m.astype(theano.config.floatX)
