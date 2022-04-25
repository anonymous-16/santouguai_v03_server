import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py

from utils.torch_utilities import (onehot_tensor, wav_2_spec)
from utils.utilities import (read_lst, read_config)
from .layers import (ConvBlock1D, init_bn, LinearBlock2D, LinearBlock1D, ConvBlock, DeepConvBlock, DecoderBlock)
from .tcn import TemporalConvNet
from conf.models import *
from .ddsp_synthesizer import DDSPSynthesizer

et = 1e-8


def predict_f0(prob, note):
	res = prob.shape[1]
	prob = torch.sigmoid(prob.detach())
	ind = torch.arange(res)[None, :, None].to(prob.device)
	prob = prob / prob.sum(1).unsqueeze(1)
	target = (ind * prob).sum(1).round().long()
	target[note == NOTES_NUM - 1] = 0
	target[target > res - 1] = 0
	target = (2**((target / F0_RES - 48) / 12)) * 440	
	return target

def predict_en(prob):
	res = prob.shape[1]
	prob = torch.sigmoid(prob.detach())
	ind = torch.arange(res)[None, :, None].to(prob.device)
	prob = prob / prob.sum(1).unsqueeze(1)
	target = (ind * prob).sum(1).round().long()
	return target


def predict(prob):
	prob = torch.sigmoid(prob.detach())
	prob[prob > 0.5] = 1
	prob[prob < 1] = 0
	return prob

def soft_predict(prob, dt=0.5):
	prob = torch.sigmoid(prob)
	prob[prob < dt] = 0
	return prob

def predict_mf0(f0_prob, note_target):
	B, N, T = note_target.shape
	f0_w = torch.sigmoid(f0_prob)
	f0_w = F.pad(f0_w[:, :-3], (0, 0, 3, 0), "constant", 0)	
	f0_w = f0_w.view(B, N, -1, T)
	_, _, R, _ = f0_w.shape
	f0_w = (f0_w * torch.arange(R).to(f0_w.device)[None, None, :, None]).sum(2) / f0_w.sum(2)
	f0_w = onehot_tensor(f0_w, R).transpose(-1, -2)
	f0_w = (f0_w * note_target[:, :, None]).flatten(1, 2)
	f0_w = F.pad(f0_w[:, 3:], (0, 0, 0, 3), "constant", 0)
	return f0_w 

class FiLMLayer(nn.Module):
	def __init__(self, rep_dim, condition_dim):
		super().__init__()

		self.gamma = LinearBlock1D(condition_dim, rep_dim)
		self.beta = LinearBlock1D(condition_dim, rep_dim)

	def forward(self, x, y):
		g = self.gamma(y)
		b = self.beta(y)
		return x * g + b

class DUnet(nn.Module):
	def __init__(self, blocks, input_size):
		super().__init__()
		encs = nn.ModuleList()
		decs = nn.ModuleList()
		in_channels = 1
		for block in blocks[:-1]:
			encs.append(ConvBlock(in_channels, block))
			in_channels=block

		self.mid_layer = ConvBlock(blocks[-2], blocks[-1])

		in_channels = blocks[-1]
		for block in list(reversed(blocks[:-1])):
			decs.append(DecoderBlock(in_channels, block, strides=(2, 2)))
			in_channels = block

		self.encs = encs
		self.decs = decs

		in_channels = in_channels * input_size
		self.output_dim = in_channels
	
	def forward(self, input):
		x = input
		if len(x.shape) == 3:
			x = x.unsqueeze(1)
		concat_x = []
		for layer in self.encs:
			x = layer(x)
			concat_x = [x] + concat_x
			x = F.avg_pool2d(x, kernel_size=(2, 2))
		x = self.mid_layer(x)
		
		for i, layer in enumerate(self.decs):
			x = layer(x, concat_x[i])

		x = x.flatten(1, 2)
		return x

class UnitModule(nn.Module):
	def __init__(self, blocks, input_size, timbre_dim, classes_num, cout_dim, dis_out):
		super().__init__()

		self.dunet = DUnet(blocks=blocks,
												input_size=input_size)

		in_channels = self.dunet.output_dim
		rep_dim = timbre_dim if dis_out else timbre_dim * classes_num
		cin_dim = timbre_dim * classes_num
		self.lower_rep_extractor = LinearBlock1D(in_channels, cout_dim)
		self.upper_rep_extractor = LinearBlock1D(in_channels, cout_dim)
		self.lower_conditioner = FiLMLayer(cout_dim, classes_num)
		self.upper_conditioner = FiLMLayer(cout_dim, classes_num)

		self.classifier = LinearBlock1D(in_channels, classes_num)
		self.classes_num = classes_num
		self.bottom_dim = in_channels
		self.timbre_dim = timbre_dim

	def forward(self, x):
		x = self.dunet(x)
		lower_rep = self.lower_rep_extractor(x)
		upper_rep = self.upper_rep_extractor(x)
		prob = self.classifier(x)
		return lower_rep, upper_rep, prob
		

	def entangle(self, x, y=None, z=None):
		print(x.shape)
		D = self.timbre_dim
		if len(x.shape) == 2:
			x = onehot_tensor(x, self.classes_num + 1).transpose(-1, -2)[:, :-1]
		upper_rep = None
		lower_rep = None
		B, N, T = x.shape
		if y is not None:
			#y = (y.view(B, -1, D, T) * x[:, :, None]).flatten(1, 2)
			lower_rep = self.lower_conditioner(y, x)
		if z is not None:
			#z = (z.view(B, -1, D, T) * x[:, :, None]).flatten(1, 2)
			upper_rep = self.upper_conditioner(z, x)
		
		if upper_rep is None and lower_rep is not None:
			return lower_rep 
		if upper_rep is not None and lower_rep is None:
			return upper_rep 
		
		return lower_rep, upper_rep


class SepModule(UnitModule):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		timbre_dim = kwargs["timbre_dim"]
		classes_num = kwargs["classes_num"]
		self.abstract_layer = LinearBlock1D(self.bottom_dim, classes_num * timbre_dim)

	def forward(self, x):
		x = self.dunet(x)
		lower_rep = self.lower_rep_extractor(x)
		upper_rep = self.upper_rep_extractor(x)
		prob = self.classifier(x)
		abt_rep = self.abstract_layer(x)
		return lower_rep, upper_rep, prob, abt_rep

	def instr_vector(self, rep, target, center=False):
		if len(rep.shape) == 3:
			B, _, T	= rep.shape
			rep = rep.view(B, self.classes_num, -1, T)
		if len(target.shape) == 4:
			rep = rep[:, None]
		target = target.unsqueeze(-2)
		target_frac = target.sum(-3, keepdim=True)
		target_frac[target_frac < 1] = 1
		ratio = target / target_frac
		rep = (rep * ratio ).sum(-3)
		
		nonsil = target.sum(-3)
		nonsil[nonsil > 0] = 1

		if center:
			frames = nonsil.sum(-1)
			rep = rep.sum(-1) / frames
		return rep, nonsil


	def shifting(self, shifting, prob, rep, lb, ub):
		B, N, T = prob.shape
		rep = rep.view(B, N, -1, T).transpose(1, -1)[:, :, :, lb:ub]
		shifting = shifting.transpose(1, -1)
		rep = torch.matmul(rep, shifting).transpose(1, -1)
		rep = F.pad(rep, (0, 0, 0, 0, lb, N - ub), "constant", 0)
		rep = self.condition_layer(rep.flatten(1, 2))
		return rep


class DDSPParameter(nn.Module):
	def __init__(self, blocks, input_size, f0s_num, energies_num, timbre_dim):
		super().__init__()

		self.dunet = DUnet(blocks=blocks,
												input_size=input_size)
		in_channels = self.dunet.output_dim
		
		self.ti_extractor = LinearBlock1D(in_channels, timbre_dim)
		self.f0_classifier = LinearBlock1D(in_channels, f0s_num)
		self.energies_classifier = LinearBlock1D(in_channels, energies_num)


	def forward(self, x):
		x = self.dunet(x)
		ti = self.ti_extractor(x)
		f0 = self.f0_classifier(x)
		en = self.energies_classifier(x)
		return ti, f0, en

class Synthesizer(nn.Module):
	def __init__(self, input_size, blocks, output_size):
		super().__init__()

		self.dunet = DUnet(blocks=blocks,
												input_size=input_size)
		in_channels = blocks[0]
		self.bottom = LinearBlock2D(in_channels, 1)

	def forward(self, x):
		B, D, T = x.shape
		x = self.dunet(x)
		x = self.bottom(x.view(B, -1, D, T)).squeeze(1)
		return x
		

class DisentanglementModel(nn.Module):
	def __init__(self):
		super(DisentanglementModel, self).__init__()
		model_name = "santouguai"
		hparams = MODEL_CONFIG[model_name]
		
		notes_num = hparams["notes_num"]
		f0_res = hparams["f0_res"]
		f0s_num = hparams["f0s_num"]
		energies_num = hparams["energies_num"]
		blocks = hparams["blocks"]
		timbre_dim = hparams["timbre_dim"]
		f0_timbre_dim = hparams["f0_timbre_dim"]
		note_timbre_dim = hparams["note_timbre_dim"]
		ddsp_timbre_dim = hparams["ddsp_timbre_dim"]

		feature_size = hparams["feature_size"]

		self.sep_module = SepModule(blocks=blocks,
															input_size=feature_size,
															classes_num=f0s_num,
															timbre_dim=f0_timbre_dim,
															cout_dim=timbre_dim,
															dis_out=False)


		self.note_module = UnitModule(blocks=blocks,
															input_size=timbre_dim,
															classes_num=notes_num,
															timbre_dim=note_timbre_dim,
															cout_dim=timbre_dim,
															dis_out=True)

		self.f0_module = DDSPParameter(blocks=blocks,
															input_size=timbre_dim,
															f0s_num=f0s_num,
															energies_num=energies_num,
															timbre_dim=ddsp_timbre_dim)

		self.spec_module = Synthesizer(input_size=timbre_dim,
																		blocks=blocks,
																		output_size=feature_size)


		self.ddsp_module = DDSPSynthesizer(hidden_size=ddsp_timbre_dim)

	def forward(self, input, mode):
			return getattr(self, mode)(input)


	#def separate(self, input):
	#	x, shifting, lb, ub = input
	#	mix_rep, mix_f0_prob, _ = self.sep_module(x)
	#	spec_in = self.sep_module.shifting(shifting, mix_f0_prob, mix_rep, lb, ub)
	#	spec = self.spec_module(spec_in)
	#	return spec, mix_f0_prob

	def separate(self, input):
		x, sep_x = input
		rep, _, f0_prob, _ = self.sep_module(x)
		_, _, sep_f0_prob, _ = self.sep_module(sep_x)
		sep_f0_target = predict(sep_f0_prob)
		spec_in = self.sep_module.entangle(x=sep_f0_target,
																				y=rep)
		spec = self.spec_module(spec_in)
		return spec, f0_prob

	def extract_queries(self, x):
		_, prob, _ = self.sep_module(x)
		return predict(prob)

	def extract_instr_rep(self, input):
		rep, target, center = input
		rep, nonsil = self.sep_module.instr_vector(rep, target, center)
		return rep, nonsil

	def extract_note_in(self, x, sep_x):
		_, mix_rep, _, _ = self.sep_module(x)
		_, _, sep_prob, _ = self.sep_module(sep_x)
		sep_target = predict(sep_prob)
		note_in = self.sep_module.entangle(x=sep_target,
																				z=mix_rep)
		return note_in


	def inference_rep(self, x):
		low_rep, up_rep, f0_prob, abt_rep = self.sep_module(x)
		#f0_prob = soft_predict(f0_prob)
		#f0_target = torch.zeros_like(f0_prob) + f0_prob
		#f0_target[f0_target > 0] = 1
		B, N, T = f0_prob.shape
		abt_rep = abt_rep.view(B, N, -1, T)
		return low_rep, up_rep, abt_rep, f0_prob

	def extract_rep(self, x):
		_, _, f0_prob, abt_rep = self.sep_module(x)
		B, N, T = f0_prob.shape
		abt_rep = abt_rep.view(B, N, -1, T)
		return abt_rep

	def disentangle(self, input):
		x_a, x_b, s_a, s_b, note_a, f0_a, en_a = input
		note_in_a = self.extract_note_in(x_a, s_a)
		note_in_b = self.extract_note_in(x_b, s_b)

		_, _, note_prob_a = self.note_module(note_in_a)
		note_rep_b, _, _ = self.note_module(note_in_b)

		f0_in_a_b = self.note_module.entangle(x=note_a, y=note_rep_b)

		ti, f0_prob, en_prob = self.f0_module(f0_in_a_b)
		
		wav = self.ddsp_module((ti, f0_a[:, None], en_a[:, None]))

		return wav, note_prob_a, f0_prob, en_prob


	def inference_spec(self, input):
		rep, target = input
		spec_in = self.sep_module.entangle(x=target, y=rep)
		spec = self.spec_module(spec_in)
		return spec

	def synthesis(self, rep, target, midi=None):
		note_in = self.sep_module.entangle(x=target, z=rep)
		print("2.", note_in.shape)
		note_rep, _, note_prob = self.note_module(note_in)
		if midi is None:
			print("3.", note_rep.shape, note_prob.shape)
			print(note_prob[0, :, 0])
			_, note = torch.max(note_prob, 1)
			print(note.shape)
			print(torch.max(note))
		else:
			note = midi
		f0_in = self.note_module.entangle(x=note, y=note_rep)
		ti, f0_prob, en_prob = self.f0_module(f0_in)
		f0_v = predict_f0(f0_prob, note)
		en_v = predict_en(en_prob)
		wav = self.ddsp_module((ti, f0_v[None, :], en_v[None, :]))
		f0_v = onehot_tensor(f0_v, f0_prob.shape[1] + 1).transpose(-1, -2)[:, :-1]
		return wav, f0_v


	def inference_mono(self, input):
		rep, target = input
		print("1.", rep.shape, target.shape)
		wav, f0_v = self.synthesis(rep, target)
		return wav, f0_v

	def inference_synthesis(self, input):
		rep, target, midi_track = input
		wav, f0_v = self.synthesis(rep, target, midi_track)
		return wav, f0_v

