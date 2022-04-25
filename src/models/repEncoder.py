import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../..'))


from src.models.layers import (LinearBlock2D, ConvBlock, DecoderBlock)
from src.conf.repEncoder import *


class RepEncoder(nn.Module):
	def __init__(self):
		super(RepEncoder, self).__init__()
		
		enc_layers = nn.ModuleList()
		dec_layers = nn.ModuleList()

		for i in range(len(ENC_CHANNELS) - 1):
			enc_layers.append(ConvBlock(in_channels=ENC_CHANNELS[i], out_channels=ENC_CHANNELS[i + 1]))
			layer = nn.ConvTranspose2d(in_channels=ENC_CHANNELS[i + 1],
									out_channels=DEC_CHANNELS[i], kernel_size=3, stride=(1, 2**(i + 1)), dilation=(1, 2**i),
									padding=(1, 0), output_padding=0, bias=False)

			dec_layers.append(layer)

		self.enc_layers = enc_layers
		self.dec_layers = dec_layers

	def forward(self, input):

		enc_layers = self.enc_layers
		dec_layers = self.dec_layers

		x = input
		out = []

		for i, layer in enumerate(enc_layers):
			x = layer(x)
			x = F.avg_pool2d(x, kernel_size=(2, 2))
			c = dec_layers[i](x)
			c = c[:, :, :, :-1]
			out.append(c)

		return out




if __name__=="__main__":
	repEncoder = RepEncoder()
	x = torch.randn([1, 1, 1024, 288])
	y = repEncoder(x)
	for o in y:
		print(o.shape)
