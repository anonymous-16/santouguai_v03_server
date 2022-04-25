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

from .layers import mlp
from .syntheiszer import UnitRender
from utils.utilities import freq_2_note, note_2_freq, freq_2_bin
from utils.torch_utilities import onehot_tensor
from feature_extraction.spec_filter import init_freq_bin_mask




class ContextRender(nn.Module):
	def __init__(self, note_distance, timbre_rep_size, hidden_size, n_layers):
		super().__init__()
		render = nn.ModuleList()
		render.append(mlp(note_distance * 2 + 1, hidden_size, n_layers, kernel_size=1))
		render.append(mlp(hidden_size * 2**(n_layers - 1), timbre_rep_size, 1, act=""))
		self.render = nn.Sequential(*render)

	def forward(self, timbre, relation_a, relation_b):
		context_a = self.render(relation_a)
		context_b = self.render(relation_b)
		a = (context_a[:, :, :, None] * context_b[:, :, None]).sum(1, keepdim=True)
		z = (F.softmax(a, -1) * timbre[:, :, None]).sum(-1)
		z = torch.sigmoid(z)
		return z
		

class Synthesizer(nn.Module):
	def __init__(self, timbre_rep_size, note_distance, notes_num, timbre_layers, sample_rate, hidden_size, n_layers, n_hamonic, n_fft):

		self.notes_num = notes_num
		self.note_distance = note_distance

	def forward(self, note, f0, f_note, b_note, duration, note_low_bound, note_up_bound):
		
