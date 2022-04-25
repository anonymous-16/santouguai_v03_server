import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py
import math

EPS = 1e-8


def init_layer(layer):
	"""Initialize a Linear or Convolutional layer. """
	nn.init.xavier_uniform_(layer.weight)

	if hasattr(layer, 'bias'):
		if layer.bias is not None:
			layer.bias.data.fill_(0.)


def init_bn(bn):
	"""Initialize a Batchnorm layer. """
	bn.bias.data.fill_(0.)
	bn.weight.data.fill_(1.)



class ConvBlock(nn.Module):
	"""A Convolutional Layer Followed by a Batchnorm Layer and a ReLU Activation Layer.

		Input : [B x in_channels x T x F]
		Output : [B x out_chanels x T x F]

		Parameters
		-----------
		in_channels : int
		out_channels : int
		momentum : float
		
	"""
	def __init__(self, in_channels, out_channels, momentum=0.01):
		super(ConvBlock, self).__init__()

		self.conv = nn.Conv2d(in_channels=in_channels,
								out_channels=out_channels,
								kernel_size=(3, 3), stride=(1, 1),
								padding=(1, 1), bias=False)

		self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
		
		self.init_weights()		

	def init_weights(self):
		init_layer(self.conv)
		init_bn(self.bn)


	def forward(self, input):
		"""
		Parameters
		----------
		input : [B x in_channels x T x F]
		
		Returns
		-------
		x : [B x out_chanels x T x F]

		"""	
		x = input
		x = F.relu_(self.bn(self.conv(x)))
		return x


class DeepConvBlock(nn.Module):
	"""2 Convolutional Layers, each of which is followed by a Batchnorm Layer and a ReLU Activation Layer.

		Input : [B x in_channels x T x F]
		Output : [B x out_chanels x T x F]

		Parameters
		-----------
		in_channels : int
		out_channels : int
		momentum : float
		
	"""

	def __init__(self, in_channels, out_channels, momentum=0.01):
		super(DeepConvBlock, self).__init__()

		self.conv1 = ConvBlock(in_channels, out_channels, momentum)
		self.conv2 = ConvBlock(out_channels, out_channels, momentum)


	def forward(self, input):
		"""

		Parameters
		----------
		input : [B x in_channels x T x F]

		Returns
		-------
			:	[B x out_chanels x T x F]
		"""
		x = input
		return self.conv2(self.conv1(x))


class LinearBlock2D(nn.Module):
	"""1 2D 1x1 Convolutional Layer with bias.
	
		Input : [B x in_channels x T x F]
		Output : [B x out_chanels x T x F]

		Parameters
		-----------
		in_channels : int
		out_channels : int
		
	"""

	def __init__(self, in_channels, out_channels):
		super(LinearBlock2D, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels,
								out_channels=out_channels,
								kernel_size=(1, 1), stride=(1, 1), bias=True)
		
		self.init_weights()


	def init_weights(self):
		init_layer(self.conv)

	def forward(self, input):
		"""
		Parameters
		----------
		input : [B x in_channels x T x F]

		Returns
		-------
		x : [B x out_chanels x T x F]	
		"""
		x = input
		x = self.conv(x)
		return x
	
class LinearBlock1D(nn.Module):
	"""1 1D 1x1 Convolutional Layer.

		Input : [B x in_channels x T]
		Output : [B x out_chanels x T]
	
		Parameters
		-----------
		in_channels : int
		out_channels : int
		bias : boolean
			default : true
			has bias if true
		
	"""

	def __init__(self, in_channels, out_channels, bias=True):
		super(LinearBlock1D, self).__init__()
		self.conv = nn.Conv1d(in_channels=in_channels,
								out_channels=out_channels,
								kernel_size=1, stride=1, bias=bias)
		
		self.init_weights()
		

	def init_weights(self):
		init_layer(self.conv)


	def forward(self, input):
		"""
		Parameters
		-----------
		input : [B x in_channels x T]	

		Returns
		-----------
		x : [B x out_chanels x T]
		"""
	
		x = input
		x = self.conv(x)
		return x
	
class EncoderBlock(nn.Module):
	"""A Convolutional Layer Followed by a Batchnorm Layer and a ReLU Activation Layer.

		Look details of the description at `ConvBlock`.
		
	"""

	def __init__(self, in_channels, out_channels, momentum = 0.01):
		super(EncoderBlock, self).__init__()

		self.conv_block = ConvBlock(in_channels, out_channels, momentum)

	def forward(self, input):
		x = input
		x = self.conv_block(x)
		#x_pool = F.avg_pool2d(x, kernel_size=self.downsample)
		return x

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, strides, momentum=0.01, deep=False):
		super(DecoderBlock, self).__init__()

		self.conv = torch.nn.ConvTranspose2d(in_channels=in_channels,
			out_channels=out_channels, kernel_size=(3, 3), stride=strides,
			padding=(0, 0), output_padding=(0, 0), bias=False)

		self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
		self.conv_block = DeepConvBlock(out_channels * 2, out_channels, momentum) if deep else ConvBlock(out_channels * 2, out_channels, momentum)
		
		self.init_weights()

		self.prune_temporal = (not strides[-2] == 1)
		
	def init_weights(self):
		init_layer(self.conv)
		init_bn(self.bn)

	def prune(self, x):
		"""Prune the shape of x after transpose convolution.
		"""
		if self.prune_temporal:
			x = x[:, :, : - 1, : - 1]
		else:
			x = x[:, :, 1 : -1, : -1]
		return x


	def forward(self, input_tensor, concat_tensor):
		x = input_tensor
		x = F.relu_(self.bn(self.conv(x)))
		x = self.prune(x)
		x = torch.cat((x, concat_tensor), dim=1)
		x = self.conv_block(x)
		return x



class DecoderBlockV2(nn.Module):
	def __init__(self, in_channels, out_channels, strides, momentum=0.01):
		super(DecoderBlockV2, self).__init__()

		self.conv = torch.nn.ConvTranspose2d(in_channels=in_channels,
			out_channels=out_channels, kernel_size=(3, 3), stride=strides,
			padding=(0, 0), output_padding=(0, 0), bias=False)

		self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
		self.conv_block = ConvBlock(out_channels, out_channels, momentum)

		self.init_weights()

		self.prune_temporal = (not strides[-2] == 1)

	def init_weights(self):
		init_layer(self.conv)
		init_bn(self.bn)

	def prune(self, x):
		"""Prune the shape of x after transpose convolution.
		"""
		if self.prune_temporal:
			x = x[:, :, : - 1, : - 1]
		else:
			x = x[:, :, 1 : -1, : -1]
		return x


	def forward(self, input_tensor):
		x = input_tensor
		x = F.relu_(self.bn(self.conv(x)))
		x = self.prune(x)
		x = self.conv_block(x)
		return x



class ConvBlock1D(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size=3, momentum=0.01):
		super(ConvBlock1D, self).__init__()

		self.conv = nn.Conv1d(in_channels=in_channels,
								out_channels=out_channels,
								kernel_size=3, stride=1,
								padding=1, bias=False)

		self.bn = nn.BatchNorm1d(out_channels, momentum=momentum)

		self.init_weights()

	def init_weights(self):
		init_layer(self.conv)
		init_bn(self.bn)

	def forward(self, input):
		x = input
		x = F.relu_(self.bn(self.conv(x)))
		return x

