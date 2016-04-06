from __future__ import division
from io import BytesIO
import math

import numpy
from PIL import Image
from six import PY3

try:
    from fuel.transformers._image import window_batch_bchw
    window_batch_bchw_available = True
except ImportError:
    window_batch_bchw_available = False
from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
from fuel import config

import math
from scipy.signal import convolve2d
from random import randint

# elastic transform function for elastic warp purpose (RandomElasticTransform Fuel transformer)
# from erniejunior/elastic_transform.py @ github
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    print x.shape
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


# Thanks to Florian Bordes for MaximumImageDimensions
class MaximumImageDimensions(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images to maximum dimensions.
    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    maximum_shape : 2-tuple
        The maximum `(height, width)` dimensions every image must have.
        Images whose height and width are larger than these dimensions
        are passed through as-is.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.
    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.
    """
    def __init__(self, data_stream, maximum_shape, resample='nearest',
            **kwargs):
        self.maximum_shape = maximum_shape
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(MaximumImageDimensions, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        max_height, max_width = self.maximum_shape
        original_height, original_width = example.shape[-2:]
        if original_height > max_height or original_width > max_width:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            im = numpy.array(im.resize((max_width, max_height))).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example

class RandomHorizontalSwap(SourcewiseTransformer, ExpectsAxisLabels):
    """Swap (lists of) images horizontally with probability of 0.5.
    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.
    """
    def __init__(self, data_stream, **kwargs):
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomHorizontalSwap, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        swap = numpy.random.randint(0,2)
        if swap:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            im = numpy.array(im.transpose(Image.FLIP_LEFT_RIGHT)).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example

class RandomElasticTransform(SourcewiseTransformer, ExpectsAxisLabels):
    """Perform some elastic transformation on images randomly.
    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.
    """
    def __init__(self, data_stream, **kwargs):
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomElasticTransform, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                self.data_stream.axis_labels[source_name],
                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        warp = numpy.random.randint(0,2)
        if warp:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im_dis = elastic_transform(im,alpha,sigma)
            im_dis = numpy.array(im_dis).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im_dis.transpose(2, 0, 1)
            else:
                example = im_dis
        return example





