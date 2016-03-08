#piece of code from https://ift6266h16.wordpress.com/class-project/getting-started/
#simple test using hades cluster

from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions
from fuel.transformers import Flatten

import theano
from theano import tensor
import numpy

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter

train = DogsVsCats(('train',), subset=slice(0, 20000))
stream = DataStream.default_stream(train,iteration_scheme=ShuffledScheme(train.num_examples, 128))
cropped_stream = RandomFixedSizeCrop(stream, (32, 32), which_sources=('image_features',))
flattened_stream = Flatten(cropped_stream, which_sources=('image_features',))
X = tensor.matrix('image_features')
T = tensor.lmatrix('targets')
W = theano.shared(numpy.random.uniform(low=-0.01, high=0.01, size=(3072, 500)), 'W')
b = theano.shared(numpy.zeros(500))
V = theano.shared(numpy.random.uniform(low=-0.01, high=0.01, size=(500, 2)), 'V')
c = theano.shared(numpy.zeros(2))
params = [W, b, V, c]
H = tensor.nnet.sigmoid(tensor.dot(X, W) + b)
Y = tensor.nnet.softmax(tensor.dot(H, V) + c)
loss = tensor.nnet.categorical_crossentropy(Y, T.flatten()).mean()
algorithm = GradientDescent(cost=loss, parameters=params,step_rule=Scale(learning_rate=0.1))
loss.name = 'loss'
extensions = [TrainingDataMonitoring([loss], every_n_batches=1),Printing(every_n_batches=1)]
main_loop = MainLoop(data_stream=flattened_stream, algorithm=algorithm,extensions=extensions)
main_loop.run()