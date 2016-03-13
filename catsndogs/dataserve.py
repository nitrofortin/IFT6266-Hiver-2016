from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, MaximumImageDimensions, Random2DRotation
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server

image_size = (256,256)
batch_size = 64
port = 5555

train = DogsVsCats(('train',), subset=slice(0, 20000))

stream = DataStream.default_stream(
    train,
    iteration_scheme=ShuffledScheme(train.num_examples, batch_size)
)

downscale_stream = MinimumImageDimensions(
	data_stream = stream,
	minimum_shape = image_size, 
	which_sources=('image_features',)
)

upscale_stream = MaximumImageDimensions(
	data_stream = downscale_stream, 
	maximum_shape = image_size, 
	which_sources=('image_features',)
)

rotated_stream = Random2DRotation(
	data_stream = upscale_stream, 
	which_sources=('image_features',)
)

scaled_stream = ScaleAndShift(
	data_stream = rotated_stream, 
	scale = 1./255, 
	shift = 0, 
	which_sources = ('image_features',)
)

data_stream = Cast(
	data_stream = scaled_stream, 
	dtype = 'float32', 
	which_sources = ('image_features',)
)

# start_server(lol, port=port)