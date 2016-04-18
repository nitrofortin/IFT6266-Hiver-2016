# Main libs
import numpy
import theano
from theano import tensor
# Fuel
from fuel.streams import ServerDataStream
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
# Code found here: https://github.com/bordesf/IFT6266/blob/master/CatsVsDogs/funtion_resize.py
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from fuel_transformers import MaximumImageDimensions
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server
# Blocks
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, MaxPooling, Flattener
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate,BinaryCrossEntropy
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.filter import VariableFilter
from blocks.roles import INPUT
from blocks_extras.extensions.plot import Plot

# Some tools
from toolz.itertoolz import interleave
from blocks_bricks import ELU

laptop = False
# Features parameters
pooling_sizes = [(2,2),(2,2),(1,1),(1,1),(2,2),(2,2),(1,1),(1,1)]
padding_sizes = [(0,0),(0,0),(1,1),(1,1),(1,1),(0,0),(0,0),(0,0)]
filter_sizes = [(11,11),(5,5),(5,5),(3,3),(3,3),(1,1),(1,1),(1,1)]
image_size = (256,256)
output_size = 2
num_epochs = 800
num_channels = 3
# num_filters = [20,40,80,160,320,640,640,2]
conv_steps = [(4,4),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
border_mode = 'valid'


# Theano variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

# Conv net model
conv_activation = [Rectifier() for _ in num_filters]

sbn = SpatialBatchNormalization()
conv_layers = []
conv_parameters = zip(filter_sizes, conv_steps, num_filters, pooling_sizes,padding_sizes)
for i, (filter_size,conv_step,num_filter,pooling_size,padding_size) in enumerate(conv_parameters):
  conv_layers.append(
  Convolutional(
    filter_size=filter_size,
    num_filters=num_filter,
    image_size = image_size,
    step=conv_step,
    border_mode=border_mode,
    name='conv_{}'.format(i)))
  conv_layers.append(
    conv_activation[i])
  conv_layers.append(
    MaxPooling(
      pooling_size=pooling_size, 
      step=pooling_size, 
      # padding=padding_size,
      name='pool_{}'.format(i)))


conv_layers = [sbn] + conv_layers 
conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_size,weights_init=Uniform(width=0.2), biases_init=Constant(0.))
conv_sequence.initialize()
out = Flattener().apply(conv_sequence.apply(x))
y_hat = Softmax().apply(out)

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), y_hat)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])
inputs = VariableFilter(roles=[INPUT])(cg.variables)
some_inputs = inputs[4:-6]
cg_dropout = apply_dropout(cg,some_inputs, 0.5)

# Data fuel
data_valid_stream = ServerDataStream(('image_features','targets'), False, port=4040)
data_train_stream = ServerDataStream(('image_features','targets'), False, port=4041)


# Blocks main_loop
algorithm = GradientDescent(cost=cost, parameters=cg_dropout.parameters, step_rule=Adam())

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

# extensions.append(Plot(
#     '8conv_ReLUs_BN_Dropout',
#     channels=[['train_error_rate', 'valid_error_rate'],
#               ['valid_cost', 'valid_error_rate2'],
#               ['train_total_gradient_norm']],server_url=host,after_epoch=True))

model = Model(cost)
# main_loop = MainLoop(algorithm=algorithm,data_stream=data_train_stream,model=model,extensions=extensions)
# main_loop.run()

