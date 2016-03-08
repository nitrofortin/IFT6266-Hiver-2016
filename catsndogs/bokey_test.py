# import theano
# a = theano.shared(3.)
# a.name = 'a'
# x = theano.tensor.scalar('data')
# cost = abs(x ** 2 - x ** a)
# cost.name = 'cost'

# import numpy
# from fuel.streams import DataStream
# from fuel.datasets import IterableDataset
# data_stream = DataStream(IterableDataset(numpy.random.rand(150).astype(theano.config.floatX)))

# from blocks.main_loop import MainLoop
# from blocks.algorithms import GradientDescent, Scale
# from blocks.extensions import FinishAfter
# from blocks.extensions.monitoring import TrainingDataMonitoring
# from blocks_extras.extensions.plot import Plot  
# main_loop = MainLoop(
# 	model=None, data_stream=data_stream,
# 	algorithm=GradientDescent(cost=cost,
# 		parameters=[a],
# 		step_rule=Scale(learning_rate=0.1)),
# 	extensions=[FinishAfter(after_n_epochs=1),
# 	TrainingDataMonitoring([cost, a], after_batch=True),
# 	Plot('YO', channels=[['cost'], ['a']],
# 		after_batch=True)])  
# main_loop.run() 



# Let's load and process the dataset
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.transformers import Flatten

# Load the training set
train = DogsVsCats(('train',), subset=slice(0, 20000))

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream.default_stream constructor will turn our
# 8-bit images into floating-point decimals in [0, 1].
stream = DataStream.default_stream(
    train,
    iteration_scheme=ShuffledScheme(train.num_examples, 128)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to take random crops of size (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    stream, (32, 32), which_sources=('image_features',))

# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
flattened_stream = Flatten(
    cropped_stream, which_sources=('image_features',))


# Create the Theano MLP
import theano
from theano import tensor
import numpy

X = tensor.matrix('image_features')
T = tensor.lmatrix('targets')

W = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(3072, 500)), 'W')
b = theano.shared(numpy.zeros(500))
V = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(500, 2)), 'V')
c = theano.shared(numpy.zeros(2))
params = [W, b, V, c]

H = tensor.nnet.sigmoid(tensor.dot(X, W) + b)
Y = tensor.nnet.softmax(tensor.dot(H, V) + c)

loss = tensor.nnet.categorical_crossentropy(Y, T.flatten()).mean()

from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks_extras.extensions.plot import Plot  
loss.name = 'loss'
main_loop = MainLoop(
	model=None, data_stream=flattened_stream,
	algorithm=GradientDescent(cost=loss,
		parameters=params,
		step_rule=Scale(learning_rate=0.1)),
	extensions=[FinishAfter(after_n_epochs=1),
	TrainingDataMonitoring([loss], after_batch=True),
	Plot('YO', channels=['loss'],
		after_batch=True)])  
main_loop.run() 