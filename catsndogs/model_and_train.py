# Conv net LeNet sytle, inspired by https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/
# and Florian Bordes conv net for course project.

# Main libs
import numpy
import theano
from theano import tensor
# Fuel
from fuel.streams import ServerDataStream
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
# Thanks to Florian Bordes for MaximumImageDimensions transformer that allows us to define maximum images size.
# Code found here: https://github.com/bordesf/IFT6266/blob/master/CatsVsDogs/funtion_resize.py
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from maxTransformerFBordes import MaximumImageDimensions
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server
# Blocks
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, MaxPooling, Flattener
from blocks.initialization import Constant, Uniform
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks_extras.extensions.plot import Plot
# Some tools
from toolz.itertoolz import interleave

# Features parameters
pooling_sizes = [(2,2),(2,2)]
filter_sizes = [(5,5),(5,5)]
# image_size = (256, 256)
image_size = (128,128)
output_size = 2

num_channels = 3
num_filters = [20, 50]
mlp_hiddens = [500]
conv_step = (1, 1)
border_mode = 'valid'

# Theano variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

# Conv net
conv_activation = [Rectifier() for _ in num_filters]
mlp_activation = [Rectifier() for _ in mlp_hiddens] + [Softmax()]

conv_parameters = zip(filter_sizes, num_filters)

conv_layers = list(interleave([
	(Convolutional(
	filter_size=filter_size,
	num_filters=num_filter,
	image_size = image_size,
	step=conv_step,
	border_mode=border_mode,
	name='conv_{}'.format(i)) for i, (filter_size, num_filter) in enumerate(conv_parameters)),
	conv_activation,
	(MaxPooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))

conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_size,weights_init=Uniform(width=0.2), biases_init=Constant(0.))
conv_sequence.initialize()
out = Flattener().apply(conv_sequence.apply(x))

top_mlp_dims = [numpy.prod(conv_sequence.get_dim('output'))] + mlp_hiddens + [output_size]
top_mlp = MLP(mlp_activation, top_mlp_dims,weights_init=Uniform(width=0.2),biases_init=Constant(0.))
top_mlp.initialize()

predict = top_mlp.apply(out)

cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict)
error_rate = error.copy(name='error_rate')
cg = ComputationGraph([cost, error_rate])


batch_size = 64
num_epochs = 100
save_to = "CatsVsDogs.pkl"

train = DogsVsCats(('train',), subset=slice(0, 20000))
valid = DogsVsCats(('train',), subset=slice(20000,25000))

train_stream = DataStream(
    train,
    iteration_scheme=ShuffledScheme(train.num_examples, batch_size)
)

downscale_train_stream = MinimumImageDimensions(
	data_stream = train_stream,
	minimum_shape = image_size, 
	which_sources=('image_features',)
)

upscale_train_stream = MaximumImageDimensions(
	data_stream = downscale_train_stream, 
	maximum_shape = image_size, 
	which_sources=('image_features',)
)

rotated_train_stream = Random2DRotation(
	data_stream = upscale_train_stream, 
	which_sources=('image_features',)
)

scaled_train_stream = ScaleAndShift(
	data_stream = rotated_train_stream, 
	scale = 1./255, 
	shift = 0, 
	which_sources = ('image_features',)
)

data_train_stream = Cast(
	data_stream = scaled_train_stream, 
	dtype = 'float32', 
	which_sources = ('image_features',)
)

valid_stream = DataStream(
    valid,
    iteration_scheme=ShuffledScheme(valid.num_examples, batch_size)
)

downscale_valid_stream = MinimumImageDimensions(
	data_stream = valid_stream,
	minimum_shape = image_size, 
	which_sources=('image_features',)
)

upscale_valid_stream = MaximumImageDimensions(
	data_stream = downscale_valid_stream, 
	maximum_shape = image_size, 
	which_sources=('image_features',)
)

rotated_valid_stream = Random2DRotation(
	data_stream = upscale_valid_stream, 
	which_sources=('image_features',)
)

scaled_valid_stream = ScaleAndShift(
	data_stream = rotated_valid_stream, 
	scale = 1./255, 
	shift = 0, 
	which_sources = ('image_features',)
)

data_valid_stream = Cast(
	data_stream = scaled_valid_stream, 
	dtype = 'float32', 
	which_sources = ('image_features',)
)
data_train_stream = scaled_train_stream
data_valid_stream = scaled_valid_stream
# data_valid_stream = ServerDataStream(('image_features','targets'), False, port=4040)
# data_train_stream = ServerDataStream(('image_features','targets'), False, port=4041)

algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))

extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              DataStreamMonitoring(
                  [cost, error_rate],
                  data_valid_stream,
                  prefix="valid"),
              TrainingDataMonitoring(
                  [cost, error_rate,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True),
              Checkpoint(save_to),
              ProgressBar(),
              Printing()]

host = 'http://localhost:5040'
extensions.append(Plot(
    'CatsVsDogs',
    channels=[['train_error_rate', 'valid_error_rate'],
              ['valid_cost', 'valid_error_rate'],
              ['train_total_gradient_norm']],server_url=host,after_epoch=True))

model = Model(cost)
main_loop = MainLoop(algorithm=algorithm,data_stream=data_train_stream,model=model,extensions=extensions)
main_loop.run()
