import os

# Errors come from conv_sequence.get_dim('output'), it returns None
print "------------------------ Blocks errors ------------------------"
os.system("python model_and_train.py")
# I got
# Traceback (most recent call last):
#   File "model_and_train.py", line 67, in <module>
#     conv_sequence.initialize()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/base.py", line 652, in initialize
#     self.allocate()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/base.py", line 614, in allocate
#     self.push_allocation_config()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/base.py", line 682, in push_allocation_config
#     self._push_allocation_config()
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/conv.py", line 498, in _push_allocation_config
#     output_shape = layer.get_dim('output')
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/conv.py", line 301, in get_dim
#     ignore_border=self.ignore_border, padding=self.padding))
#   File "/Users/admin/anaconda2/lib/python2.7/site-packages/theano/tensor/signal/pool.py", line 192, in out_shape
#     if ds[0] == st[0]:
# TypeError: 'int' object has no attribute '__getitem__'
