#Thanks to Thrandis @ github for class basic structure

import numpy
import theano

from blocks.initialization import NdarrayInitialization

class GlorotInitialization(NdarrayInitialization):
    u"""Initialize the parameters using the Xavier initialization scheme.
    This initialization only works for fully connected layers
    (2D matrices) and is intended for use with tanh activations. More
    details about it can be found in [AISTATS10]_.
    .. [AISTATS10] Xavier Glorot and Yoshua Bengio, *Understanding the
        difficulty of training deep feedforward neural networks*, AISTATS
        (2010), pp. 249-256.
    """
    def generate(self, rng, shape):
        if len(shape) == 2:
            input_size, output_size = shape
            high = numpy.sqrt(6) / numpy.sqrt(input_size + output_size)
        elif len(shape) == 4:
            output_size, input_size, chan_x, chan_y = shape
            high = numpy.sqrt(6) / numpy.sqrt(input_size + output_size + chan_x + chan_y)
        m = rng.uniform(-high, high, size=shape)
        return m.astype(theano.config.float32)
