import os

# Errors come from conv_sequence.initialize(), conv layers act like NoneType
print "------------------------ Blocks errors ------------------------"
os.system("python model_and_train.py")
# I got
# AttributeError                            Traceback (most recent call last)
# /Users/admin/Documents/HEC/Hiver 2016/Algorithmes d'apprentissage/catsndogs/catsndogs/model_and_train.py in <module>()
#      62
#      63 conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_shape)
# ---> 64 conv_sequence.initialize()
#      65 out = Flattener().apply(conv_sequence.apply(x))
#      66

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/base.pyc in initialize(self)
#     654             self.push_initialization_config()
#     655         for child in self.children:
# --> 656             child.initialize()
#     657         self._initialize()
#     658         self.initialized = True

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/base.pyc in initialize(self)
#     655         for child in self.children:
#     656             child.initialize()
# --> 657         self._initialize()
#     658         self.initialized = True
#     659

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/bricks/conv.pyc in _initialize(self)
#     110         if self.use_bias:
#     111             W, b = self.parameters
# --> 112             self.biases_init.initialize(b, self.rng)
#     113         else:
#     114             W, = self.parameters

# AttributeError: 'NoneType' object has no attribute 'initialize'
