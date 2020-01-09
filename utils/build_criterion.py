"""
Helper function for creating Loss
Maintain a complete list of all supported Loss implemented

Author: Jianxiong Cai
"""

import torch.nn as nn

def build_criterion(opts):
	"""
	:param opts: run-time options (arguments)
	:return: criterion
	"""

	criterion = None
	if opts.loss == "CrossEntropyLoss":
		criterion = nn.CrossEntropyLoss(reduction='mean')
	else:
		raise RuntimeError("Unrecognized name when creating network: {}".format(opts.dataloader))

	return criterion
