from torch.utils.data import Dataset
from prefetch_generator import BackgroundGenerator
from dataset.urmp.urmp_sample import UrmpDataset
import numpy as np

from utils.utilities import one_hot
from conf.urmp import *
from conf.sample import *

dt = 1e-7

def pad(x, pad_value):
	if x.shape[0] < MAX_SOURCES_NUM:
		pad_len = MAX_SOURCES_NUM - x.shape[0]
		if len(x.shape) == 2:
			return np.pad(x, ((0, pad_len), (0, 0)), 'constant', constant_values=(pad_value, pad_value))
		elif len(x.shape) == 3:
			return np.pad(x, ((0, pad_len), (0, 0), (0, 0)), 'constant', constant_values=(pad_value, pad_value))
		else:
			assert False
	return x

def stack(x, axis=0, merge=False):
	output = []
	for line in x:
		if len(output) == 0:
			for i in range(len(line)):
				output.append([])
		for i, d in enumerate(line):
			output[i].append(d)
	output = [np.stack(o, axis) for o in output]
	if merge:
		output = [o.sum(axis) for o in output]
	return output

class DataSampler(Dataset):
	
	def __init__(self, sample_mode="next"):
		super(DataSampler, self).__init__()
		self.rng = np.random.RandomState(1234)

		self.dataset = UrmpDataset(self.rng)
		self.samples_num = self.dataset.get_samples_num()
		self.sample_mode = sample_mode

	def __iter__(self):
		return BackgroundGenerator(super().__iter__())

	def __getitem__(self, index = 0):
		return getattr(self, self.sample_mode)(index)


	def sample(self, instrs_num):
		instrs = self.sample_unit(instrs_num)
		instr_index = np.zeros([instrs_num])
		while np.sum(instr_index) == 0:
			instr_index = self.rng.randint(0, 2, size=(instrs_num,))	
		instr_mask = np.zeros([MAX_SOURCES_NUM], dtype=np.int16)
		instr_mask[:instrs_num] = instr_index
		return instrs, instr_index, instr_mask

	def sample_unit(self, instrs_num, included=[], excluded=[], repeat=False):
		instrs = [] + included
		while len(instrs) < instrs_num:
			r = self.rng.randint(0, len(self._instr_tag))
			if (repeat or r not in instrs) and (r not in excluded):
				if (np.array(instrs) == r).sum() <= 3:
					instrs.append(r)
		
		pos = []
		neg = []
		for ind in instrs:
			if ind == instrs[0]:
				pos.append(ind)
			else:
				neg.append(ind)
		return pos + neg, len(pos)

	def sample_mixture(self, index):
		wavs, notes, onsets, instrs = self.dataset.next(index, mode="normal")
	
		res = {}
		for i, instr in enumerate(instrs):
			if not instr in res:
				res[instr] = {"wav" : [], "note" : [], "onset" : []}
			res[instr]["wav"].append(wavs[i])
			res[instr]["note"].append(notes[i])
			res[instr]["onset"].append(onsets[i])
		
		def merge(x):
			x = sum(x)
			x[x > 1] = 1
			return x

		wav = []
		note = []
		onset = []
		for instr in res:
			wav.append(sum(res[instr]["wav"]))
			note.append(merge(res[instr]["note"]))
			onset.append(merge(res[instr]["onset"]))
		note = np.stack(note, 0)
		onset = np.stack(onset, 0)
		return wav, note, onset, instrs

	def sample_nonsil(self, instr):
		bk_note = []
		while len(bk_note) < 1:
			wavs, notes, instrs = self.dataset.next(instr, mode="nonsil")	
		
			pos_note = []
			bk_note = []

			for i, v in enumerate(instrs):
				if v == instr:
					pos_note.append(notes[i])
				else:
					bk_note.append(notes[i])

		def merge(x):
			x = sum(x)
			x[x > 1] = 1
			return x

		mixture = sum(wavs)
		pos = merge(pos_note)
		remain = []
		while len(bk_note) < MAX_SOURCES_NUM - 1:
			bk_note = bk_note + bk_note
		bk_note = bk_note[:MAX_SOURCES_NUM - 1]
		note = np.stack([pos] + bk_note, 0)
		return mixture, note

	def separate_without_dis(self, index):
		wav, note, onset, _ = self.sample_mixture(index)
		mixture = sum(wav)
		sep = wav[0]
		def merge(x):
			x = x.sum(0)
			x[x > 1] = 1
			return x
		batch = (mixture, sep, note[0], merge(note), merge(onset))
		return batch
		
	def clustering(self, index):
		wav, note, onset, instrs = self.sample_mixture(index)
		nonsil_mix, nonsil_note = self.sample_nonsil(instrs[0])
		mixture = sum(wav)
		batch = (mixture, nonsil_mix, note[0], note.sum(0), nonsil_note)
		return batch


	def __len__(self):
		return SAMPLES_NUM
		#return self.samples_num

	def get_len(self):
		return self.__len__()

