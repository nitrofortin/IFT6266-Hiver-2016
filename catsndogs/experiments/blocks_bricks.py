"""Some of the simplest individual bricks."""
# Must be enhance to allow setting custom alpha parameter at initialization.
import logging

from theano import tensor

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.interfaces import Activation, Feedforward, Initializable
from blocks.bricks.interfaces import Random  # noqa

from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans

logger = logging.getLogger(__name__)

class ELU(Activation):	
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
    	self.alpha = 1.0
        return tensor.switch(input_ >= 0, input_, self.alpha*(tensor.exp(input_) - 1))
