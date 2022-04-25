import os
import sys
import time
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import math
from scipy import signal

from .synthesizer import Synthesizer
from .layers import ffn, dffn
from .tcn import TemporalConvNet 
from utils.utilities import freq_2_note, note_2_freq, freq_2_bin
from utils.torch_utilities import onehot_tensor, positional_encoding
from feature_extraction.spec_filter import init_freq_bin_mask
from conf.sample import *

et = 1e-7

class ContextGenerator(nn.Module):
	def __init__(self):
		super().__init__()
		nothing = 0

	def forward(self, rep, duration_mask, position_mask, low_duration, up_duration):
		kernel = torch.FloatTensor([[[0.5, 0.5]]]).to(rep.device)
		res = 0.
		B, D, T = rep.shape
		rep = rep.transpose(1, 2).flatten(0, 1)[:, None]
		densed_rep = F.conv1d(rep, kernel, stride=2)
		pot = NOTE_RES // 2
		for i in range(0, NOTE_RES - low_duration):
			win_len = NOTE_RES - i - 1
			if win_len == pot:
				rep = densed_rep
				densed_rep = F.conv1d(rep, kernel, stride=2) if i < NOTE_RES - 2 else None
				d = rep
				pot //= 2
			else:
				pad_len = win_len - pot
				d = torch.cat([rep[:, :, :2* pad_len], densed_rep[:, :, pad_len:]], -1)

			if win_len < up_duration:
				res += (d.reshape([B, T, -1]).transpose(1, 2) * duration_mask[:, win_len - 1 : win_len, :] * position_mask[:, : win_len, :]).sum(1, keepdim=True)
		
		return res

class ContextEmbedding(nn.Module):
	def __init__(self, prop_size, n_layers, hidden_size, freq_bin):
		super().__init__()
		
		context_len = NOTE_RES
		context_layers = CONTEXT_LAYERS
		self.context_len = context_len
		self.context_layers = context_layers

		rep_size = 16

		self.f0Render = ffn(1, rep_size, None, 1, bias=False)
		self.contextRender = ffn(prop_size, context_len * rep_size, hidden_size, n_layers, context_layers=context_layers, squeeze=2)
		self.specRender = ffn(1 + 2 + rep_size, freq_bin // 2, hidden_size, n_layers, squeeze=2, context_layers=n_layers-1)
		
		pitchRender = nn.ModuleList()
		for i in range(NOTES_NUM):
			pitchRender.append(ffn(freq_bin // 2, freq_bin, None, 1, bias=False))
		self.prop_size = prop_size
		self.pitchRender = pitchRender
		self.rep_size = rep_size

	def forward(self, pitch, context, pos, note, f0_w, low_bound, up_bound, inference=False):



		context_layers = self.context_layers
		prop_size = self.prop_size
		rep_size = self.rep_size
		#context = self.contextRender(context)

		context_len = self.context_len

		#B = context.shape[0]
		#context = context.reshape([B, rep_size, context_len, -1])		
		#a = (context * pos[:, None]).sum(2)

		a = self.f0Render(f0_w)
		prop = torch.cat([torch.rand_like(pos[:, :1]), a, note], 1)
		rep = self.specRender(prop)
		spec = 0.
		if rep.shape[-1] < pitch.shape[-1]:
			lpad = (pitch.shape[-1] - rep.shape[-1]) // 2
		for i in range(low_bound, up_bound):
			spec += self.pitchRender[i](rep) * pitch[:, i : i + 1, lpad:-lpad]
		spec	+= self.pitchRender[-1](rep) * pitch[:, -1:, lpad:-lpad]
		spec = F.softmax(spec, 1)
		return spec

class InstrumentRender(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers, rep_size):
		super().__init__()



		self.decayEmbedding = ffn(in_size=input_size,
																out_size=rep_size,
																hidden_size=hidden_size,
																n_layers=n_layers,
																bias=True,
																squeeze=2)

	def forward(self, x):
		decay_coefficient = self.decayEmbedding(x)
		decay_coefficient = torch.sigmoid(decay_coefficient)
		return decay_coefficient


class ContextSynthesizer(nn.Module):
	def __init__(self, instruments_num, notes_num, hidden_size, n_layers, n_fft, rep_size):
		super().__init__()

		freq_bin = n_fft // 2 + 1

		self.contextEmbedding = ContextEmbedding(freq_bin=freq_bin,
																							prop_size=4 + 4,
																							n_layers=n_layers, 
																							hidden_size=hidden_size)


		
		self.instrumentRender = InstrumentRender(input_size=8,
																							hidden_size=hidden_size,
																							n_layers=n_layers,
																							rep_size=freq_bin)


		self.notes_num = notes_num


	def encode_instrument(self, decay_id):
		decay_coefficient = self.instrumentRender(decay_id)
		return decay_coefficient



	def encode_decay(self, first_frame, spec, decay_coefficient, en, inference=False):

		if not inference:
			teacher_spec = first_frame[:, :, :-2] * decay_coefficient[:, :, 1:-1] + spec[:, :, 1:-1]
			infer_spec = teacher_spec * en[:, :, 1:-1] / (en[:, :, :-2] * decay_coefficient[:, :, 1:-1] + en[:, :, 1:-1])
		else:
			return spec[:, :, 1:-1]
			if first_frame is None:
				first_frame = torch.zeros_like(spec[:, :, 0])
				en[:, :, 0] = 0
			specs = []
			specs.append(first_frame)
			DECAY = 0.
			for i in range(1, spec.shape[-1] - 1):
				specs.append((specs[-1] * decay_coefficient[:, :, i] + spec[:, :, i]) * en[:, :, i] / (en[:, :, i - 1] * decay_coefficient[:, :, i] + en[:, :, i]))
			infer_spec = torch.stack(specs[1:], -1)

		return infer_spec


	def forward(self, first_frame, note, f0_w, duration_w, pos, context,
									decay_vec, low_bound, up_bound, inference=False, en=None):

		et = 1e-7

		pitch = onehot_tensor(note[:, 0], self.notes_num).transpose(1, 2)

		note = note[:, 1:]

		decay_id = torch.cat([decay_vec, note, duration_w], 1)
		decay_coefficient = self.encode_instrument(decay_id)

		context = torch.cat([context[:, :4], decay_vec], 1)
		spec	= self.contextEmbedding(pitch, context, pos, note, f0_w, low_bound, up_bound, inference)
	
		ori_en = first_frame.sum(1, keepdim=True) if not inference else en.sum(1, keepdim=True)


		if ori_en.shape[-1] > spec.shape[-1]:
			lpad = (ori_en.shape[-1] - spec.shape[-1]) // 2
			ori_en = ori_en[:, :, lpad : -lpad]
		spec = spec * ori_en
		#spec = self.encode_decay(first_frame, spec, decay_coefficient, ori_en, inference)
		
		return spec



