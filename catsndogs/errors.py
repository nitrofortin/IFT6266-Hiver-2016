import os

print "------------------------ Fuel errors ------------------------"

# Errors come from start_server function when passing data stream as input
os.system("python dataserve.py")

# I got
# INFO:fuel.server:server started
# Traceback (most recent call last):
#   File "dataserve.py", line 49, in <module>
#     start_server(data_stream, port=port)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/server.py", line 123, in start_server
#     data = next(it)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/six.py", line 558, in next
#     return type(self).__next__(self)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.py", line 32, in __next__
#     data = self.data_stream.get_data()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 138, in get_data
#     data = next(self.child_epoch_iterator)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/six.py", line 558, in next
#     return type(self).__next__(self)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.py", line 32, in __next__
#     data = self.data_stream.get_data()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 138, in get_data
#     data = next(self.child_epoch_iterator)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/six.py", line 558, in next
#     return type(self).__next__(self)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.py", line 32, in __next__
#     data = self.data_stream.get_data()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 138, in get_data
#     data = next(self.child_epoch_iterator)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/six.py", line 558, in next
#     return type(self).__next__(self)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.py", line 32, in __next__
#     data = self.data_stream.get_data()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 138, in get_data
#     data = next(self.child_epoch_iterator)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/six.py", line 558, in next
#     return type(self).__next__(self)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.py", line 32, in __next__
#     data = self.data_stream.get_data()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 151, in get_data
#     return self.transform_batch(data)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 289, in transform_batch
#     data=batch, method=self.transform_source_batch)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.py", line 250, in _apply_sourcewise_transformation
#     data[i] = method(data[i], source_name)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/image.py", line 212, in transform_source_batch
#     return [self._example_transform(im, source_name) for im in batch]
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/image.py", line 233, in _example_transform
#     im = Image.fromarray(im)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/PIL/Image.py", line 2153, in fromarray
#     raise TypeError("Cannot handle this data type")
# TypeError: Cannot handle this data type

# In this one, I included fuel steps because I was not able to build anything from data server aside.
# Errors come from ConvolutionnalSequence
print "------------------------ Blocks errors ------------------------"
os.system("python model_and_train.py")
# I got
# Traceback (most recent call last):
#   File "model_and_train.py", line 65, in <module>
#     conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_shape)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/base.py", line 842, in lazy_init
#     return init(**kwargs)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/conv.py", line 459, in __init__
#     application_methods=application_methods, **kwargs)
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/sequences.py", line 29, in __init__
#     if not (app.brick in seen or seen.add(app.brick))]
# AttributeError: 'function' object has no attribute 'brick'
