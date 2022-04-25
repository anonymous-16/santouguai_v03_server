import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

from .layers import (LinearBlock1D, ConvBlock1D, DecoderBlock1D)


class RepUnet1D(nn.Module):
	def __init__(self, enc_chs, mid_chs, dec_chs, bottom_chs):
		super(RepUnet1D, self).__init__()
		
		enc_layers = nn.ModuleList()
		dec_layers = nn.ModuleList()

		for i in range(len(enc_chs) - 1):
			in_chs = enc_chs[i]
			out_chs = enc_chs[i + 1]
			enc_layers.append(ConvBlock1D(in_channels=in_chs, out_channels=out_chs))

		mid_layer = ConvBlock1D(in_channels=mid_chs[0], out_channels=mid_chs[1])

		for i in range(len(dec_chs) - 1):
			dec_layers.append(DecoderBlock1D(in_channels=dec_chs[i], out_channels=dec_chs[i + 1], strides=2))

		in_chs = bottom_chs[0]
		out_chs = bottom_chs[1]
		bottom_layer = LinearBlock1D(in_chs, out_chs) 

		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.mid_layer = mid_layer
		self.bottom_layer = bottom_layer

	def forward(self, input):

		enc_layers = self.enc_layers
		dec_layers = self.dec_layers

		x = input
		concat_tensors = []

		for layer in enc_layers:
			x = layer(x)
			concat_tensors.append(x)
			x = F.avg_pool1d(x, kernel_size=2)

		x = self.mid_layer(x)

		for i, layer in enumerate(dec_layers):
			x = layer(x, concat_tensors[- 1 - i])

		out = self.bottom_layer(x)

		return out




if __name__=="__main__":
	repUnet = RepUnet()
	x = torch.randn([1, 1, 1024, 288])
	repUnet = RepUnet()
	y = repUnet(x)
	print(x.shape, y.shape)
