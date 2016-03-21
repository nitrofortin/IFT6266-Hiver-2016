# According to Sermanet et al. 'OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks'
# Require too much memory
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
from fuel_transformers import MaximumImageDimensions
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server
# Blocks
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, MaxPooling, Flattener
from blocks.initialization import Constant, Uniform
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks_extras.extensions.plot import Plot
# Some tools
from toolz.itertoolz import interleave

laptop = True
# Features parameters
pooling_sizes = [(2,2),(2,2),(1,1),(1,1),(2,2)]
filter_sizes = [(11,11),(5,5),(3,3),(3,3),(3,3)]
image_size = (231,231)
output_size = 2
num_epochs = 500
num_channels = 3
num_filters = [96,256,512,1024,1024]
mlp_hiddens = [3072,4096]
conv_step = (1, 1)
border_mode = 'valid'

# Theano variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

# Conv net model
conv_activation = [Rectifier() for _ in num_filters]
mlp_activation = [Rectifier() for _ in mlp_hiddens] + [Softmax()]

conv_parameters = zip(filter_sizes, num_filters)
sbn = SpatialBatchNormalization()
conv_layers = list(interleave([
  (Convolutional(
  filter_size=filter_size,
  num_filters=num_filter,
  step=conv_step,
  border_mode=border_mode,
  name='conv_{}'.format(i)) for i, (filter_size, num_filter) in enumerate(conv_parameters)),
  conv_activation,
  (MaxPooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))

conv_layers = [sbn] + conv_layers
conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_size,weights_init=Uniform(width=0.2), biases_init=Constant(0.))
conv_sequence.initialize()
out = Flattener().apply(conv_sequence.apply(x))

top_mlp_dims = [numpy.prod(conv_sequence.get_dim('output'))] + mlp_hiddens + [output_size]
top_mlp = MLP(mlp_activation, top_mlp_dims,weights_init=Uniform(0,0.2),biases_init=Constant(0.))
top_mlp.initialize()

predict = top_mlp.apply(out)

cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])
# inputs = VariableFilter(roles=[INPUT])(cg.variables)
# cg_dropout = apply_dropout(cg,inputs, 0.5)

# Data fuel
data_valid_stream = ServerDataStream(('image_features','targets'), False, port=4040)
data_train_stream = ServerDataStream(('image_features','targets'), False, port=4041)


# Blocks main_loop
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())

extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              DataStreamMonitoring(
                  [cost, error_rate, error_rate2],
                  data_valid_stream,
                  prefix="valid"),
              TrainingDataMonitoring(
                  [cost, error_rate,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True),
             # Checkpoint(save_to),
              ProgressBar(),
              Printing()]

if laptop:
  host = 'http://localhost:5040'
else:
  host = 'http://hades.calculquebec.ca:5050'

extensions.append(Plot(
    'CatsVsDogs',
    channels=[['train_error_rate', 'valid_error_rate'],
              ['valid_cost', 'valid_error_rate2'],
              ['train_total_gradient_norm']],server_url=host,after_epoch=True))
model = Model(cost)
main_loop = MainLoop(algorithm=algorithm,data_stream=data_train_stream,model=model,extensions=extensions)
main_loop.run()