import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

#sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from .layers import (LinearBlock2D, LinearBlock1D, ConvBlock1D, DecoderBlock1D)
from .cRepUnet import CRepUnet
from conf.separator import *
from conf.feature import *


class CCNN(nn.Module):
	def __init__(self, channels, condition_dim):
		super(CCNN, self).__init__()
		net = nn.ModuleList()
		fiLM_w_layers = nn.ModuleList()
		fiLM_b_layers = nn.ModuleList()
		
		for i in range(len(channels) - 2):
			net.append(ConvBlock1D(channels[i], channels[i + 1]))
			fiLM_w_layers.append(LinearBlock1D(condition_dim, channels[i + 1]))
			fiLM_b_layers.append(LinearBlock1D(condition_dim, channels[i + 1]))
		
		self.net = net
		self.fiLM_w_layers = fiLM_w_layers
		self.fiLM_b_layers = fiLM_b_layers
		self.bottom = LinearBlock1D(channels[-2], channels[-1])
		
	def forward(self, x, condition):
		for i, layer in enumerate(self.net):
			x = layer(x)
			w = self.fiLM_w_layers[i](condition)
			b = self.fiLM_b_layers[i](condition)
			x = x * w + b
		x = self.bottom(x)
		return x

class Separator(nn.Module):
	def __init__(self):
		super(Separator, self).__init__()
		self.repSeparator = CRepUnet()
		self.timbreExtractor = CCNN(TIMBRE_CHANNELS, CONDITION_DIM)
		self.energyExtractor = CCNN(ENERGY_CHANNELS, CONDITION_DIM)
		self.pitchExtractor = CCNN(PITCH_CHANNELS, CONDITION_DIM)
		self.noteTranscriptor = CCNN(NOTE_CHANNELS, CONDITION_DIM)
		self.f0Transcriptor = CCNN(F0_CHANNELS, CONDITION_DIM)
		#self.transcriptor = CRepUnet1D(output_dim=NOTES_NUM + 1)	
		
	def forward(self, mixture, query):
		h = self.repSeparator(mixture, query)
		h = h.squeeze(1)
		query = query.squeeze(-2)
		query = torch.tanh(query)
		ti = self.timbreExtractor(h, query)
	
		t_query = torch.tanh(ti)
		en = self.energyExtractor(h, t_query)
		p = self.pitchExtractor(h, t_query)
		note = self.noteTranscriptor(p, t_query)
		f0 = self.f0Transcriptor(p, t_query)
		en_ratio = torch.sigmoid(en)
		f0 = F.softplus(f0)
		return ti, note, f0, en_ratio


if __name__=="__main__":
	separator = Separator()
	mixture = torch.randn([1, 1, 1024, 288])
	query = torch.randn([1, 128, 1, 1])
	ti, note, f0, en_ratio = separator(mixture, query)
	print(ti.shape, note.shape, f0.shape, en_ratio.shape)
