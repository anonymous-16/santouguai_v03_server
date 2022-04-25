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


et = 1e-8

def predict_f0(note_target, f0_prob):
	f0_prob = torch.sigmoid(f0_prob)
	B, NN, T = note_target.shape
	f0_prob = f0_prob.reshape(B, NN, -1, T)

	_, _, res, _ = f0_prob.shape
	a_c = torch.arange(res)[None, None, :, None].to(f0_prob.device)
	frac = f0_prob.sum(2)
	frac[frac == 0] = 1
	mono_f0 = (f0_prob * a_c).sum(2) / frac
	mono_f0 = onehot_tensor(mono_f0.round(), res).transpose(-1, -2)

	f0_target = note_target[:, :, None] * mono_f0

	return note_target, f0_target.flatten(1, 2)

def predict(prob):
	prob = torch.sigmoid(prob.detach())
	prob[prob > 0.5] = 1
	prob[prob < 1] = 0
	return prob

def soft_predict(prob, dt=0.5):
	prob = torch.sigmoid(prob)
	prob[prob < dt] = 0
	return prob


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
	def __init__(self, blocks, input_size, deep=False, in_chs=1):
		super().__init__()
		encs = nn.ModuleList()
		decs = nn.ModuleList()
		in_channels = in_chs
		for block in blocks[:-1]:
			if deep:
				encs.append(DeepConvBlock(in_channels, block))
			else:
				encs.append(ConvBlock(in_channels, block))
			in_channels=block

		self.mid_layer = DeepConvBlock(blocks[-2], blocks[-1]) if deep else ConvBlock(blocks[-2], blocks[-1])

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



class SepModule(nn.Module):
	def __init__(self, blocks, input_size, timbre_dim, abstract_timbre_dim, cout_dim, note_classes_num, ctx_num):
		super().__init__()

		self.network = DUnet(blocks=blocks, input_size=input_size)

		in_channels = self.network.output_dim
		self.rep_extractor = LinearBlock1D(in_channels, note_classes_num * timbre_dim)
		self.condition_layer = LinearBlock1D(note_classes_num * timbre_dim, cout_dim)

		self.abstract_layer = LinearBlock1D(timbre_dim, abstract_timbre_dim)
		self.classifier_layer = LinearBlock1D(abstract_timbre_dim, abstract_timbre_dim + 1)	
	
		self.note_classifier = LinearBlock1D(in_channels, note_classes_num)
		self.onset_classifier = LinearBlock1D(in_channels, note_classes_num)

		self.note_classes_num = note_classes_num
		self.bottom_dim = in_channels
		self.timbre_dim = timbre_dim


	def forward(self, x):
		x = self.network(x)
		rep = self.rep_extractor(x)
		B, _, T = rep.shape

		NN = self.note_classes_num
		note_prob = self.note_classifier(x)
		onset_prob = self.onset_classifier(x)
		abt_rep = self.abstract_layer(rep.reshape(B, NN, -1, T).flatten(0, 1)).reshape(B, NN, -1, T)

		return rep, note_prob, onset_prob, abt_rep

	def instr_vector(self, rep, target, center=False):
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

	def instr_activate(self, rep, center):
		dc = self.classifier_layer(center.transpose(1, 2))
		w = dc[:, :-1]
		b = dc[:, -1:]
		res = torch.matmul(rep.transpose(-1, -2).flatten(1, 2), w) + b
		return res

	def entangle(self, x, y):
		D = self.timbre_dim
		if len(x.shape) == 2:
			x = onehot_tensor(x, self.note_classes_num + 1).transpose(-1, -2)[:, :-1]
		B, N, T = x.shape
		y = (y.view(B, -1, D, T) * x[:, :, None]).flatten(1, 2)
		#if onset is not None:
		#	y = (torch.randn_like(y).reshape(B, self.note_classes_num, -1, T) * onset[:, :, None]).flatten(1, 2) + y
		y = self.condition_layer(y)
		return y


	def shifting(self, shifting, prob, rep, lb, ub):
		B, N, T = prob.shape
		_, M, _ = onset.shape
		rep = (torch.randn_like(rep).reshape(B, M, -1, T) * onset[:, :, None]).flatten(1, 2) + rep
		rep = rep.view(B, N, -1, T).transpose(1, -1)[:, :, :, lb:ub]
		shifting = shifting.transpose(1, -1)
		rep = torch.matmul(rep, shifting).transpose(1, -1)
		rep = F.pad(rep, (0, 0, 0, 0, lb, N - ub), "constant", 0)
		rep = self.condition_layer(rep.flatten(1, 2))
		return rep

	def query(self, x, note_target, center):
		x = self.network(x)
		note_prob, f0_prob = self.query_separator(x, note_target, center)
		return x, note_prob, f0_prob

class Synthesizer(nn.Module):
	def __init__(self, input_size, blocks, output_size):
		super().__init__()

		self.network = DUnet(blocks=blocks,
												input_size=input_size)


		in_channels = blocks[0]
		self.bottom = LinearBlock2D(in_channels, 1)

	def forward(self, x):
		B, D, T = x.shape
		x = self.network(x)
		x = self.bottom(x.view(B, -1, D, T)).squeeze(1)
		return x
		

class DisentanglementModel(nn.Module):
	def __init__(self):
		super(DisentanglementModel, self).__init__()
		model_name = "santouguai"
		hparams = MODEL_CONFIG[model_name]
		
		notes_num = hparams["notes_num"]
		ctx_num = hparams["ctx_num"]
		blocks = hparams["blocks"]
		timbre_dim = hparams["timbre_dim"]
		note_timbre_dim = hparams["note_timbre_dim"]
		ab_timbre_dim = hparams["ab_timbre_dim"]

		feature_size = hparams["feature_size"]

		self.sep_module = SepModule(blocks=blocks,
																input_size=feature_size,
																timbre_dim=note_timbre_dim,
																abstract_timbre_dim=ab_timbre_dim,
																cout_dim=feature_size,
																note_classes_num=notes_num,
																ctx_num=ctx_num)


		self.spec_module = Synthesizer(input_size=timbre_dim,
																		blocks=blocks,
																		output_size=feature_size)

	def forward(self, input, mode):
			return getattr(self, mode)(input)

	def get_generation_parameters(self):
		return list(self.note_module.parameters()) + list(self.f0_module.parameters())

	def separate(self, input):
		x, shifting, onset, lb, ub = input
		mix_rep, note_prob, onset_prob,_ = self.sep_module(x)
		spec_in = self.sep_module.shifting(shifting, mix_f0_prob, mix_rep, onset, lb, ub)
		spec = self.spec_module(spec_in)
		return spec, mix_f0_prob, note_prob, onset_prob

	def separate_without_dis(self, input):
		x, note = input
		mix_rep, note_prob, onset_prob, _ = self.sep_module(x)
		B, M, T = note_prob.shape
		spec_in = self.sep_module.entangle(note, mix_rep)
		spec = self.spec_module(spec_in)
		return spec, note_prob, onset_prob


	def instr_activate(self, input):
		rep, center = input
		return self.sep_module.instr_activate(rep, center)


	def extract_queries(self, x):
		_, prob, _ = self.sep_module(x)
		return predict(prob)

	def extract_instr_rep(self, input):
		rep, target, center = input
		rep, nonsil = self.sep_module.instr_vector(rep, target, center)
		return rep, nonsil


	def inference_rep(self, x):
		rep, note_prob, onset_prob, abt_rep = self.sep_module(x)
		return rep, abt_rep, note_prob, onset_prob
		#f0_target, note_target, cluster_target, onset_target = predict_f0(f0_prob, note_prob, cluster_prob, onset_prob)
		#return rep, abt_rep, f0_target, note_target, cluster_target, onset_target

	def extract_rep(self, x):
		_, _, _, abt_rep = self.sep_module(x)
		return abt_rep


	def inference_dc(self, center):
		dc = self.sep_module.classifier_layer(center[None, :, :].transpose(1, 2))
		return dc.squeeze(0).transpose(0, 1)

	def inference_spec(self, input):
		rep, note_target = input
		#rep, _, _, _, _, _ = self.sep_module(x)
		#_, f0_target = predict_f0(note_target, f0_prob)
		spec_in = self.sep_module.entangle(x=note_target, y=rep)
		spec = self.spec_module(spec_in)
		#_, f0_prob, note_prob, _, _, _ = self.sep_module(spec)
		#note_target = predict(note_prob)
		#_, f0_target = predict_f0(note_target, f0_prob)
		#spec_in = self.sep_module.entangle(x=f0_target, y=rep)
		#spec = self.spec_module(spec_in)
		#note_target = f0_target.view(1, NOTES_NUM - 1, -1, f0_target.shape[-1]).sum(2)
		#note_target[note_target > 1] = 1
		return spec, note_target




