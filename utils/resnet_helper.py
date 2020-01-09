"""
Helper function to build resnet block using torchvision resnet modules

Adapted from torchvision/models/resnet.py Resnet._make_layer
"""

from torch import nn

import torchvision.models.resnet as resnet

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def build_resnet_layer(inplanes, planes, blocks, stride=1, block= resnet.BasicBlock, norm_layer=nn.BatchNorm2d,
					   dilate=False, previous_dilation = 1, groups = 1, base_width=64):
	"""
	Build one resnet layer
	:param block: torchvision.models.resnet.BasicBlock
	:param inplanes: input number of channel
	:param planes: output number of channel
	:param blocks: Number of resnet block in this layer
	:param stride:
	:param dilate:
	:param previous_dilation:
		the dilation to use (first layer is always 1, successive layer should get the value from previous layer)
	:return: (model, new_dilation)
		model: the trained model
		new_dilation:
			the dilation used for second resblock and following
			When building next layer, it should be use it as initial value.
	"""
	downsample = None
	dilation_new = previous_dilation
	if dilate:
			dilation_new *= stride
			stride = 1
	if stride != 1 or inplanes != planes * block.expansion:
			downsample = nn.Sequential(
					conv1x1(inplanes, planes * block.expansion, stride),
					norm_layer(planes * block.expansion),
			)

	layers = []
	layers.append(block(inplanes, planes, stride, downsample, groups,
											base_width, previous_dilation, norm_layer))
	inplanes = planes * block.expansion
	for _ in range(1, blocks):
			layers.append(block(inplanes, planes, groups=groups,
													base_width=base_width, dilation=dilation_new,
													norm_layer=norm_layer))

	return nn.Sequential(*layers), dilation_new