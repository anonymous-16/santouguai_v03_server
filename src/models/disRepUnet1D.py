import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import numpy as np

from .layers import (LinearBlock1D, ConvBlock1D, DecoderBlock1D)


class UnitDecoder(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, stride, cat=True):
		super(UnitDecoder, self).__init__()

		net = nn.ModuleList()
		net.append(DecoderBlock1D(in_ch, mid_ch, strides=2, cat=cat))
		if stride > 1:
			net.append(nn.Upsample(scale_factor=stride, mode='nearest'))
		net.append(LinearBlock1D(mid_ch, out_channels=out_ch))
		self.net = net

	def forward(self, x, concate_tensor):
		x = self.net[0](x, concate_tensor)
		y = self.net[1](x) if len(self.net) == 3 else x
		z = self.net[-1](x)
		x = self.net[-1](y)
		return x, z

class FiLMLayer(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(FiLMLayer, self).__init__()
		self.net = ConvBlock1D(in_channels=out_ch, out_channels=out_ch)
		self.w = LinearBlock1D(in_ch, out_channels=out_ch)
		self.b = LinearBlock1D(in_ch, out_channels=out_ch)

	def forward(self, x, condition):
		w = self.w(condition)
		b = self.b(condition)
		x = x * w + b
		x = self.net(x)
		return x


class DisRepUnet1D(nn.Module):
	def __init__(self, enc_chs, mid_chs, dec_chs, bottom_chs, notes_num):
		super(DisRepUnet1D, self).__init__()
		
		enc_layers = nn.ModuleList()
		dec_layers = nn.ModuleList()
		dis_layers = nn.ModuleList()
		condition_layers = nn.ModuleList()

		for i in range(len(enc_chs) - 1):
			in_chs = enc_chs[i]
			out_chs = enc_chs[i + 1]
			enc_layers.append(ConvBlock1D(in_channels=in_chs, out_channels=out_chs))


		mid_layer = ConvBlock1D(in_channels=mid_chs[0], out_channels=mid_chs[1])
		step = 2**(len(enc_chs) - 2)
		dis_layers.append(UnitDecoder(in_ch=mid_chs[0], mid_ch=mid_chs[1], out_ch=mid_chs[1] * 2, stride=step, cat=False))	
		condition_layers.append(FiLMLayer(mid_chs[1] * 2, mid_chs[0]))
		step *= 2
		
		embedding_dim = mid_chs[1] * 2
		for i in range(len(dec_chs) - 1):
			step //= 2
			if i < len(dec_chs) - 2:
				dec_layers.append(DecoderBlock1D(in_channels=dec_chs[i], out_channels=dec_chs[i + 1], strides=2))
			embedding_dim += dec_chs[i + 1] * 2	
			condition_layers.append(FiLMLayer(embedding_dim , dec_chs[i + 1]))
			dis_layers.append(UnitDecoder(in_ch=dec_chs[i], mid_ch=dec_chs[i + 1], out_ch=dec_chs[i + 1] * 2, stride=step, cat=True))	
	
			if i == len(dec_chs) - 3:
				note_layer = UnitDecoder(in_ch=dec_chs[i], mid_ch=dec_chs[i + 1], out_ch=notes_num, stride=step, cat=True)

			if i == len(dec_chs) - 2:
				f0_layer = UnitDecoder(in_ch=dec_chs[i], mid_ch=dec_chs[i + 1], out_ch=1, stride=step, cat=True)
				en_layer = UnitDecoder(in_ch=dec_chs[i], mid_ch=dec_chs[i + 1], out_ch=1, stride=step, cat=True)

		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.dis_layers = dis_layers
		self.mid_layer = mid_layer
		self.f0_layer = f0_layer
		self.en_layer = en_layer
		self.note_layer = note_layer
		self.condition_layers = condition_layers

	def forward(self, input, condition):

		enc_layers = self.enc_layers
		dec_layers = self.dec_layers
		dis_layers = self.dis_layers
		mid_layer = self.mid_layer
		note_layer = self.note_layer
		f0_layer = self.f0_layer
		en_layer = self.en_layer
		condition_layers = self.condition_layers

		x = input
		concat_tensors = []


		for i, layer in enumerate(enc_layers):
			x = layer(x)
			x = condition_layers[- i - 1](x, condition[- i - 1]) if condition[- i - 1] is not None else x
			concat_tensors.append(x)	
			x = F.avg_pool1d(x, kernel_size=2)

		reps = []
		un_pool_reps = []

		rep, un_pool_rep = dis_layers[0](x, None)
		reps.append(F.softmax(rep, 1))
		un_pool_reps.append(F.softmax(un_pool_rep, 1))
		x = mid_layer(x)

		dis_layers = dis_layers[1:]
		for i, layer in enumerate(dis_layers):
			
			rep, un_pool_rep = layer(x, concat_tensors[- 1 - i])
			reps.append(F.softmax(rep, 1))
			un_pool_reps.append(F.softmax(un_pool_rep, 1))
		
			if i == len(dis_layers) - 2:
				note, _ = note_layer(x, concat_tensors[- 1 - i])
			if i == len(dis_layers) - 1:
				f0, _ = f0_layer(x, concat_tensors[- 1 - i])
				en, _ = en_layer(x, concat_tensors[- 1 - i])

			if i < len(dis_layers) - 1:
				x = dec_layers[i](x, concat_tensors[- 1 - i])

		out = torch.cat(reps, 1)
		f0 = torch.sigmoid(f0)
		en = torch.sigmoid(en)
		return out, un_pool_reps, note, f0, en




if __name__=="__main__":
	repUnet = RepUnet()
	x = torch.randn([1, 1, 1024, 288])
	repUnet = RepUnet()
	y = repUnet(x)
	print(x.shape, y.shape)
