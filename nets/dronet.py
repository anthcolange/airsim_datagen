"""
Extend DroNet to take stack of images

Author: Jianxiong Cai
"""

from torch import nn
from utils.resnet_helper import build_resnet_layer

class DroNetExt(nn.Module):
	"""
	Extended Version of DroNet

	Note: For some reason, DroNet didn't have activation after first conv
	The implementation here has the activation layer (ReLU)
	"""
	def __init__(self, in_channel):
		super(DroNetExt, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		# self.softmax = nn.Softmax()

		self.conv_1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=2, padding=2)
		self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

		self.resblock_1, dilation = build_resnet_layer(32, 64, 1, stride=2)
		self.resblock_2, dilation = build_resnet_layer(64, 64, 1, stride=2, previous_dilation=dilation)
		self.resblock_3, dilation = build_resnet_layer(64, 128, 1, stride=2, previous_dilation=dilation)

		self.droupout_1 = nn.Dropout(0.5)
		self.fc_1 = nn.Linear(7 * 7 * 128, 2)

	def forward(self, x):
		x = self.conv_1(x)
		x = self.relu(x)
		x = self.maxpool_1(x)
		x = self.resblock_1(x)
		x = self.resblock_2(x)
		x = self.resblock_3(x)

		x = x.view(-1, 7 * 7 * 128)
		x = self.droupout_1(x)
		x = self.fc_1(x)
		# x = self.softmax(x)

		return x


