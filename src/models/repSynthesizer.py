import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from src.models.disRepUnet1D import DisRepUnet1D
from src.models.synthesizer import Synthesizer

from src.conf.repEncoder import *

class RepSynthesizer(nn.Module):
	def __init__(self):
		super(RepSynthesizer, self).__init__()


		self.repEnc = DisRepUnet1D(ENC_CHANNELS, MID_CHANNEL, DEC_CHANNELS, BOTTOM_CHANNEL, NOTES_NUM)
		self.synthesizer = Synthesizer()
		self.timbreChs = DEC_CHANNELS

	def tr_loss(self, ti, pos_ti, neg_tis):
		ch = 0
		timbreChs = self.timbreChs
		tr_losses = []
		margin = 0.5
		for i, tch in enumerate(timbreChs):
			if i >= len(neg_tis):
				break
			tch *= 2
			tr_losses.append(F.relu(((ti[:, : ch + tch] - pos_ti[:, : ch + tch])**2).sum(1) / (i + 1) \
											- ((ti[:, : ch + tch] - neg_tis[i][:, : ch + tch])**2).sum(1) / (i + 1) + margin))
			ch += tch
		return tr_losses

	def encode_reps(self, rep, depth=1):
		reps = []
		d = 0
		timbreChs = self.timbreChs
		for i, tch in enumerate(timbreChs):
			if i >= depth:
				reps.append(None)
				continue
			d += timbreChs[i]
			reps.append(torch.cat(rep[:i + 1], 1))
		return reps

	def forward(self, spec, condition, note=None, f0=None, mode="est_gt"):
		ti, reps, est_note, est_f0, est_en = self.repEnc(spec, condition)

		if mode == "ti":
			return ti, reps, est_note, est_f0, est_en
		
		if mode == "est":
			c = est_note.shape[1]
			est_note = est_note.argmax(1)
			est_note = F.one_hot(est_note, c).transpose(2, 1)
			print(est_note.shape)
			print(f0[0, 0, 100:200])
			print(est_f0[0, 0, 100:200])
			out_spec = self.synthesizer(ti, est_note, est_f0)
		else:
			out_spec = self.synthesizer(ti, note, f0)
		
		if mode == "est_ti":
			return out_spec, ti, reps, est_note, est_f0, est_en
		return out_spec, est_en

if __name__=="__main__":
	spec = torch.randn([1, 1024, 288])
	f0 = torch.randn([1, 1, 288])
	note = torch.randn([1, 89, 288])

	repSynthesizer = RepSynthesizer()
	condition = repSynthesizer.encode_reps(None, 0)	
	print(len(condition))
	ti, reps, _, _, _ = repSynthesizer(spec, condition, None, None, "ti")
	condition = repSynthesizer.encode_reps(reps, 2)
	print(len(condition))
	print("----------")
	out, ti, _, est_note, est_f0, est_en = repSynthesizer(spec, condition, note, f0,  mode="est_ti")
