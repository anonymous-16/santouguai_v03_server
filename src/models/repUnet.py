import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

from .layers import (LinearBlock2D, LinearBlock1D, ConvBlock, DecoderBlock)


class RepUnet(nn.Module):
	def __init__(self, enc_chs, mid_chs, dec_chs, bottom_chs, input_size, flatten):
		super(RepUnet, self).__init__()
		
		enc_layers = nn.ModuleList()
		dec_layers = nn.ModuleList()

		for i in range(len(enc_chs) - 1):
			enc_layers.append(ConvBlock(in_channels=enc_chs[i], out_channels=enc_chs[i + 1]))

		mid_layer = ConvBlock(in_channels=mid_chs[0], out_channels=mid_chs[1])

		for i in range(len(dec_chs) - 1):
			dec_layers.append(DecoderBlock(in_channels=dec_chs[i], out_channels=dec_chs[i + 1], strides=(2, 2)))

		bottom_layer = LinearBlock1D(bottom_chs[0] * input_size, bottom_chs[1])

		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.mid_layer = mid_layer
		self.bottom_layer = bottom_layer
		self.flatten = flatten

	def forward(self, input):

		enc_layers = self.enc_layers
		dec_layers = self.dec_layers

		x = input
		concat_tensors = []

		for layer in enc_layers:
			x = layer(x)
			concat_tensors.append(x)
			x = F.avg_pool2d(x, kernel_size=(2, 2))

		x = self.mid_layer(x)

		for i, layer in enumerate(dec_layers):
			x = layer(x, concat_tensors[- 1 - i])

		x = x.flatten(1, 2)
		out = self.bottom_layer(x)

		return out


if __name__=="__main__":
	repUnet = RepUnet()
	x = torch.randn([1, 1, 1024, 288])
	repUnet = RepUnet()
	y = repUnet(x)
	print(x.shape, y.shape)
