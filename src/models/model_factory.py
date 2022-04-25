import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from src.utils.utilities import (read_lst, read_config)
from src.models.models import DisentanglementModel

from src.conf.feature import *

et = 1e-8

class ModelFactory(nn.Module):

	def __init__(self, model_name):
		super(ModelFactory, self).__init__()

		if model_name in ['AMT', 'AMTBaseline']:
			network = AMTBaseline()
		elif model_name in ['MSS', 'MSSBaseline']:
			network = MSSBaseline()
		elif model_name in ['MSS-AMT', 'MultiTaskBaseline']:
			network = MultiTaskBaseline()
		elif model_name in ['MSI', 'MSI-DIS', 'DisentanglementModel']:
			network = DisentanglementModel()
	
		self.network = network

	def wav2spec(self, input):
		channels_num = input.shape[-2]

		def spectrogram(input):
			(real, imag) = self.stft(input)
			spec = (real ** 2 + imag ** 2) ** 0.5
			return spec

		spec_list = []

		for channel in range(channels_num):
			spec = spectrogram(input[:, channel, :])
			spec_list.append(spec)

		spec = torch.cat(spec_list, 1)[:, :, :, :-1]
		return spec

	def forward(self, input, mode=None):
		if mode == "wav2spec":
			spec = self.wav2spec(input)
			return spec
		return self.network(input) if mode is None else self.network(input, mode)
		

if __name__ == '__main__':
	model_name = 'MSI-DIS'
	model = ModelFactory(model_name).cuda()

	query_spec = torch.randn(16, 1, 288, 128).cuda()
	mix_spec = torch.randn(16, 1, 288, 1024).cuda()
	another_mix_spec = torch.randn(16, 1, 288, 1024).cuda()
	hQuery = model(query_spec, 'query')
	print(hQuery.shape)
	args = (mix_spec, another_mix_spec, hQuery)
	est_spec, note_prob, f0_prob= model(args, 'transfer')
	
