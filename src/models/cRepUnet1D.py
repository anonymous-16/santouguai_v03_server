import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

from .layers import (LinearBlock2D, LinearBlock1D, ConvBlock1D, DecoderBlock1D)
from conf.cRepUnet1D import *


class CRepUnet1D(nn.Module):
	def __init__(self, input_dim=None, output_dim=None):
		super(CRepUnet1D, self).__init__()
		
		enc_layers = nn.ModuleList()
		dec_layers = nn.ModuleList()
		fiLM_w_layers = nn.ModuleList()
		fiLM_b_layers = nn.ModuleList()

		for i in range(len(ENC_CHANNELS) - 1):
			in_chs = ENC_CHANNELS[i] if i > 0 or input_dim is None else input_dim
			out_chs = ENC_CHANNELS[i + 1]
			enc_layers.append(ConvBlock1D(in_channels=in_chs, out_channels=out_chs))
			fiLM_w_layers.append(LinearBlock1D(CONDITION_DIM, out_chs))
			fiLM_b_layers.append(LinearBlock1D(CONDITION_DIM, out_chs))

		mid_layer = ConvBlock1D(in_channels=MID_CHANNEL[0], out_channels=MID_CHANNEL[1])
		fiLM_w_layers.append(LinearBlock1D(CONDITION_DIM, MID_CHANNEL[1]))
		fiLM_b_layers.append(LinearBlock1D(CONDITION_DIM, MID_CHANNEL[1]))

		for i in range(len(DEC_CHANNELS) - 1):
			dec_layers.append(DecoderBlock1D(in_channels=DEC_CHANNELS[i], out_channels=DEC_CHANNELS[i + 1], strides=2))
			fiLM_w_layers.append(LinearBlock1D(CONDITION_DIM, DEC_CHANNELS[i + 1]))
			fiLM_b_layers.append(LinearBlock1D(CONDITION_DIM, DEC_CHANNELS[i + 1]))


		in_chs = BOTTOM_CHANNEL[0]
		out_chs = BOTTOM_CHANNEL[1] if output_dim is None else output_dim
		bottom_layer = LinearBlock1D(in_chs, out_chs) 


		self.fiLM_w_layers = fiLM_w_layers
		self.fiLM_b_layers = fiLM_b_layers
		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.mid_layer = mid_layer
		self.bottom_layer = bottom_layer

	def forward(self, input, condition):

		enc_layers = self.enc_layers
		dec_layers = self.dec_layers

		x = input
		concat_tensors = []

		cid = 0
		for layer in enc_layers:
			x = layer(x)
			w = self.fiLM_w_layers[cid](condition)
			b = self.fiLM_b_layers[cid](condition)
			#print(x.shape, w.shape, b.shape)
			x = x * w + b
			concat_tensors.append(x)
			x = F.avg_pool1d(x, kernel_size=2)
			cid += 1
		x = self.mid_layer(x)	
		w = self.fiLM_w_layers[cid](condition)
		b = self.fiLM_b_layers[cid](condition)
		x = x * w + b

		cid += 1
		

		for i, layer in enumerate(dec_layers):
			x = layer(x, concat_tensors[- 1 - i])
			w = self.fiLM_w_layers[cid](condition)
			b = self.fiLM_b_layers[cid](condition)
			x = x * w + b

			cid += 1
		out = self.bottom_layer(x)

		return out




if __name__=="__main__":
	repUnet = RepUnet()
	x = torch.randn([1, 1, 1024, 288])
	repUnet = RepUnet()
	y = repUnet(x)
	print(x.shape, y.shape)
