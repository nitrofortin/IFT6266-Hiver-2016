import os

# Errors come from mainloop()
print "------------------------ Blocks errors ------------------------"
os.system("python model_and_train.py")
# I got
# ERROR:blocks.main_loop:Error occured during training.

# Blocks will attempt to run `on_error` extensions, potentially saving data, before exiting and reraising the error. Note that the usual `after_training` extensions will *not* be run. The original error will be re-raised and also stored in the training log. Press CTRL + C to halt Blocks immediately.
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# /Users/admin/Documents/HEC/Hiver 2016/Algorithmes d'apprentissage/catsndogs/catsndogs/model_and_train.py in <module>()
#     189 model = Model(cost)
#     190 main_loop = MainLoop(algorithm,data_stream=data_train_stream,model=model,extensions=extensions)
# --> 191 main_loop.run()

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/main_loop.pyc in run(self)
#     195                     logger.error("Error occured when running extensions." +
#     196                                  error_in_error_handling_message)
# --> 197                 reraise_as(e)
#     198             finally:
#     199                 self._restore_signal_handlers()

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/utils/__init__.pyc in reraise_as(new_exc)
#     256     new_exc.__cause__ = orig_exc_value
#     257     new_exc.reraised = True
# --> 258     six.reraise(type(new_exc), new_exc, orig_exc_traceback)
#     259
#     260

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/main_loop.pyc in run(self)
#     181                     self.status['batch_interrupt_received'] = False
#     182                 with Timer('training', self.profile):
# --> 183                     while self._run_epoch():
#     184                         pass
#     185             except TrainingFinish:

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/main_loop.pyc in _run_epoch(self)
#     228                 return False
#     229             self.status['epoch_started'] = True
# --> 230             self._run_extensions('before_epoch')
#     231         with Timer('epoch', self.profile):
#     232             while self._run_iteration():

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/main_loop.pyc in _run_extensions(self, method_name, *args)
#     261             for extension in self.extensions:
#     262                 with Timer(type(extension).__name__, self.profile):
# --> 263                     extension.dispatch(CallbackName(method_name), *args)
#     264
#     265     def _check_finish_training(self, level):

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/extensions/__init__.pyc in dispatch(self, callback_invoked, *from_main_loop)
#     336             if (callback_name == callback_invoked and
#     337                     predicate(self.main_loop.log)):
# --> 338                 self.do(callback_invoked, *(from_main_loop + tuple(arguments)))
#     339
#     340     @staticmethod

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/extensions/monitoring.pyc in do(self, callback_name, *args)
#      77         """Write the values of monitored variables to the log."""
#      78         logger.info("Monitoring on auxiliary data started")
# ---> 79         value_dict = self._evaluator.evaluate(self.data_stream)
#      80         self.add_records(self.main_loop.log, value_dict.items())
#      81         logger.info("Monitoring on auxiliary data finished")

# /Users/admin/anaconda2/lib/python2.7/site-packages/blocks/monitoring/evaluators.pyc in evaluate(self, data_stream)
#     328         self.initialize_aggregators()
#     329         if self._accumulate_fun is not None:
# --> 330             for batch in data_stream.get_epoch_iterator(as_dict=True):
#     331                 self.process_batch(batch)
#     332         else:

# /Users/admin/anaconda2/lib/python2.7/site-packages/six.pyc in next(self)
#     556
#     557         def next(self):
# --> 558             return type(self).__next__(self)
#     559
#     560     callable = callable

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.pyc in __next__(self)
#      30             data = self.data_stream.get_data(next(self.request_iterator))
#      31         else:
# ---> 32             data = self.data_stream.get_data()
#      33         if self.as_dict:
#      34             return dict(zip(self.data_stream.sources, data))

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.pyc in get_data(self, request)
#     136         if request is not None:
#     137             raise ValueError
# --> 138         data = next(self.child_epoch_iterator)
#     139
#     140         if self.produces_examples != self.data_stream.produces_examples:

# /Users/admin/anaconda2/lib/python2.7/site-packages/six.pyc in next(self)
#     556
#     557         def next(self):
# --> 558             return type(self).__next__(self)
#     559
#     560     callable = callable

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.pyc in __next__(self)
#      30             data = self.data_stream.get_data(next(self.request_iterator))
#      31         else:
# ---> 32             data = self.data_stream.get_data()
#      33         if self.as_dict:
#      34             return dict(zip(self.data_stream.sources, data))

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.pyc in get_data(self, request)
#     136         if request is not None:
#     137             raise ValueError
# --> 138         data = next(self.child_epoch_iterator)
#     139
#     140         if self.produces_examples != self.data_stream.produces_examples:

# /Users/admin/anaconda2/lib/python2.7/site-packages/six.pyc in next(self)
#     556
#     557         def next(self):
# --> 558             return type(self).__next__(self)
#     559
#     560     callable = callable

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.pyc in __next__(self)
#      30             data = self.data_stream.get_data(next(self.request_iterator))
#      31         else:
# ---> 32             data = self.data_stream.get_data()
#      33         if self.as_dict:
#      34             return dict(zip(self.data_stream.sources, data))

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.pyc in get_data(self, request)
#     136         if request is not None:
#     137             raise ValueError
# --> 138         data = next(self.child_epoch_iterator)
#     139
#     140         if self.produces_examples != self.data_stream.produces_examples:

# /Users/admin/anaconda2/lib/python2.7/site-packages/six.pyc in next(self)
#     556
#     557         def next(self):
# --> 558             return type(self).__next__(self)
#     559
#     560     callable = callable

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/iterator.pyc in __next__(self)
#      30             data = self.data_stream.get_data(next(self.request_iterator))
#      31         else:
# ---> 32             data = self.data_stream.get_data()
#      33         if self.as_dict:
#      34             return dict(zip(self.data_stream.sources, data))

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.pyc in get_data(self, request)
#     149             return self.transform_example(data)
#     150         else:
# --> 151             return self.transform_batch(data)
#     152
#     153     def transform_example(self, example):

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.pyc in transform_batch(self, batch)
#     287     def transform_batch(self, batch):
#     288         return self._apply_sourcewise_transformation(
# --> 289             data=batch, method=self.transform_source_batch)
#     290
#     291

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/__init__.pyc in _apply_sourcewise_transformation(self, data, method)
#     248         for i, source_name in enumerate(self.data_stream.sources):
#     249             if source_name in self.which_sources:
# --> 250                 data[i] = method(data[i], source_name)
#     251         return tuple(data)
#     252

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/image.pyc in transform_source_batch(self, batch, source_name)
#     210                                 self.data_stream.axis_labels[source_name],
#     211                                 source_name)
# --> 212         return [self._example_transform(im, source_name) for im in batch]
#     213
#     214     def transform_source_example(self, example, source_name):

# /Users/admin/anaconda2/lib/python2.7/site-packages/fuel/transformers/image.pyc in _example_transform(self, example, _)
#     231             else:
#     232                 im = example
# --> 233             im = Image.fromarray(im)
#     234             width, height = im.size
#     235             multiplier = max(1, min_width / width, min_height / height)

# /Users/admin/anaconda2/lib/python2.7/site-packages/PIL/Image.pyc in fromarray(obj, mode)
#    2151         except KeyError:
#    2152             # print typekey
# -> 2153             raise TypeError("Cannot handle this data type")
#    2154     else:
#    2155         rawmode = mode

# TypeError: Cannot handle this data type

# Original exception:
# 	TypeError: Cannot handle this data type
