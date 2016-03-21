# Getting more informations about network strucutre. For debugging purpose.
for i, param in enumerate(model.get_parameter_values().items()):
	print i, param[1].shape

