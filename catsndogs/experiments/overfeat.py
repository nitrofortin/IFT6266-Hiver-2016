# Inspired by AlexNet architecture
# From paper 'Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks, NIPS 2012'
# Link to the paper: http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012
# Some remarks:
# - Our model does not allow communication between convolutional sequences
# - Batch normalization between each convolutional layers
# - Dropout (p = 0.5) between each fully-connected layers
# - Adam optimization method is used(link to the paper: http://arxiv.org/pdf/1412.6980.pdf)
# - Stochastic Gradient Descent as learning algorithm is used
# - Build with Blocks and Fuel
# ---------------------------------------------------------------
# Main libraries
# ---------------------------------------------------------------
import numpy
import theano
from theano import tensor

# ---------------------------------------------------------------
# Fuel objects
# ---------------------------------------------------------------
from fuel.streams import ServerDataStream
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server

# ---------------------------------------------------------------
# Blocks objects
# ---------------------------------------------------------------
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, MaxPooling, Flattener
from blocks.initialization import Constant, Uniform
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import INPUT
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks_extras.extensions.plot import Plot

# ---------------------------------------------------------------
# Some other tools
# ---------------------------------------------------------------
from toolz.itertoolz import interleave
from operator import itemgetter
#from blocks_bricks import ELU
from glorotinit import GlorotInitialization

laptop = False
# ---------------------------------------------------------------
# Convolutional parameters
# ---------------------------------------------------------------
pooling_sizes = [(2,2),(2,2),(2,2),(2,2)]
filter_sizes = [(5,5),(5,5),(5,5),(4,4)]
num_filters = [10,20,30,40]
conv_step = (1, 1)
num_channels = 3

# ---------------------------------------------------------------
# General and Fully-connected parameters
# ---------------------------------------------------------------
image_size = (256,256)
output_size = 2
num_epochs = 500
learning_rate = 0.01
mlp_hiddens = [1000]
border_mode = 'valid'

# Theano variables
x = tensor.tensor4('image_features')
x2 = x.copy(name='x2')
y = tensor.lmatrix('targets')

# Conv net model
conv_activation = [Rectifier() for _ in num_filters]
mlp_activation = [Rectifier() for _ in mlp_hiddens] + [Softmax()]

conv_parameters = zip(filter_sizes, num_filters,pooling_sizes)

# ---------------------------------------------------------------
# 1st convolutional sequence
# ---------------------------------------------------------------

conv_layers = []
for i, (filter_size,num_filter,pooling_size) in enumerate(conv_parameters):
  conv_layers.append(SpatialBatchNormalization(name='sbn_{}_0'.format(i)))
  conv_layers.append(
  Convolutional(
    filter_size=filter_size,
    num_filters=num_filter,
    step=conv_step,
    border_mode=border_mode,
    name='conv_{}_0'.format(i)))
  conv_layers.append(conv_activation[i])
  conv_layers.append(MaxPooling(pooling_size, name='pool_{}_0'.format(i)))


# ---------------------------------------------------------------
# 2nd convolution sequence
# ---------------------------------------------------------------

conv_layers2 = []
for i, (filter_size,num_filter,pooling_size) in enumerate(conv_parameters):
  conv_layers2.append(SpatialBatchNormalization(name='sbn_{}_1'.format(i)))
  conv_layers2.append(
  Convolutional(
    filter_size=filter_size,
    num_filters=num_filter,
    step=conv_step,
    border_mode=border_mode,
    name='conv_{}_1'.format(i)))
  conv_layers2.append(conv_activation[i])
  conv_layers2.append(MaxPooling(pooling_size, name='pool_{}_1'.format(i)))


# ---------------------------------------------------------------
# Building both sequences and merge them by tensor.concatenate
# ---------------------------------------------------------------

conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_size,weights_init=Uniform(width=0.2), biases_init=Constant(0.), name='conv_sequence_0')
conv_sequence2 = ConvolutionalSequence(conv_layers2, num_channels, image_size=image_size,weights_init=Uniform(width=0.2), biases_init=Constant(0.), name='conv_sequence_1')

conv_sequence.initialize()
conv_out1 = Flattener(name='flattener_0').apply(conv_sequence.apply(x))
conv_out2 = Flattener(name='flattener_1').apply(conv_sequence2.apply(x2))
conv_out = tensor.concatenate([conv_out1,conv_out2],axis=1)

top_mlp_dims = [2*numpy.prod(conv_sequence.get_dim('output'))] + mlp_hiddens + [output_size]
top_mlp = MLP(mlp_activation, top_mlp_dims,weights_init=GlorotInitialization(),biases_init=Constant(0.))
top_mlp.initialize()

predict = top_mlp.apply(conv_out)

# ---------------------------------------------------------------
# Building computational graph
# ---------------------------------------------------------------

cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])
inputs = VariableFilter(roles=[INPUT])(cg.variables)
linear_inputs_index = [-10,-8,6]
linear_inputs = list(itemgetter(*linear_inputs_index)(inputs))
cg_dropout = apply_dropout(cg,linear_inputs, 0.5)

# ---------------------------------------------------------------
# Set ports listeners for Fuel data servers
# ---------------------------------------------------------------
data_valid_stream = ServerDataStream(('image_features','targets'), False, port=3040)
data_train_stream = ServerDataStream(('image_features','targets'), False, port=3041)


# ---------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------
#algorithm = GradientDescent(cost=cost, parameters=cg_dropout.parameters, step_rule=Adam())
algorithm = GradientDescent(cost=cost, parameters=cg_dropout.parameters, step_rule=Scale(learning_rate=learning_rate))

save_to = 'Glorot__overfeat_4conv_1full_bn.pkl'
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
             Checkpoint(save_to),
              ProgressBar(),
              Printing()]

if laptop:
  host = 'http://localhost:5040'
else:
  host = 'http://hades.calculquebec.ca:5050'

# ---------------------------------------------------------------
# Bokeh live plotting extension
# ---------------------------------------------------------------

extensions.append(Plot(
    '4conv_overfeat_1full_ReLU_bn_dropout',
    channels=[['train_error_rate', 'valid_error_rate'],
              ['valid_cost', 'valid_error_rate2'],
              ['train_total_gradient_norm']],server_url=host,after_epoch=True))

model = Model(cost)
main_loop = MainLoop(algorithm=algorithm,data_stream=data_train_stream,model=model,extensions=extensions)
# ---------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------
main_loop.run()


# ---------------------------------------------------------------
# Test the model and print results in some CSV file
# ---------------------------------------------------------------

from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers.image import MinimumImageDimensions
from fuel_transformers import MaximumImageDimensions
from fuel.transformers import Cast, ScaleAndShift

batch_size =1

test = DogsVsCats(('test',))
test_scheme = SequentialScheme(examples=test.num_examples, batch_size=batch_size)

test_stream = DataStream(
    test,
    iteration_scheme=test_scheme
)

downscale_test_stream = MinimumImageDimensions(
  data_stream = test_stream,
  minimum_shape = image_size, 
  which_sources=('image_features',)
)

upscale_test_stream = MaximumImageDimensions(
  data_stream = downscale_test_stream, 
  maximum_shape = image_size, 
  which_sources=('image_features',)
)

scaled_test_stream = ScaleAndShift(
  data_stream = upscale_test_stream, 
  scale = 1./255, 
  shift = 0, 
  which_sources = ('image_features',)
)

data_test_stream = Cast(
  data_stream = scaled_test_stream, 
  dtype = 'float32', 
  which_sources = ('image_features',)
)
test_x =tensor.tensor4('image_features')
predict_function = theano.function(inputs=[test_x], outputs=top_mlp.apply(Flattener().apply(conv_sequence.apply(test_x))))

import csv
csvfile = csv.writer(open("test_pred_overfeat.csv",'wb'))
for i,test_image in enumerate(data_test_stream.get_epoch_iterator()):
    prediction = predict_function(test_image[0])[0]
    isadog = numpy.argmax(prediction)
    csvfile.writerow([str(i+1), str(isadog)])


