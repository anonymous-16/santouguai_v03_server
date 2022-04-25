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

		self.dataset = {}
		self._instr_tag = []
		file_lst = []
		st = 0
		for i, instr in enumerate(SEEN_INSTRUMENTS):
			instr_vector = np.zeros([INSTR_GROUPS_NUM])
			group = SEEN_INSTRUMENTS[instr]
			instr_vector[group] = 1
			self.dataset[instr] = UrmpDataset(self.rng, instr, instr_vector)
			file_lst += [self.dataset[instr].get_file_path()]
			self._instr_tag += [instr]

		self.samples_num = 0
		for tag in self.dataset:
			self.samples_num += self.dataset[tag].get_samples_num()

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

	def sample_mixture(self, index, mode, merge=False, repeat=True, all_source=False, included=[], excluded=[]):
		sources_num = self.rng.randint(1, MAX_SOURCES_NUM + 1) if not all_source else MAX_SOURCES_NUM
		mixture, sep_num = self.sample_unit(sources_num, included=included, excluded=excluded, repeat=repeat)

		batches = []
		for i, instr_id in enumerate(mixture):
			tag = self._instr_tag[instr_id]
			batch = self.dataset[tag].next(index, mode=mode)
			batches.append(batch)

		if merge:
			mixture_id = {}
			for i, instr in enumerate(mixture):
				if str(instr) not in mixture_id:
					mixture_id[str(instr)] = [batches[i]]
				else:
					mixture_id[str(instr)].append(batches[i])

			mixture = []
			batches = []
			for instr in mixture_id:
				mixture.append(int(instr))
				batches.append(stack(mixture_id[instr], merge=True))
		return batches, sources_num, sep_num, mixture


	def separate_without_dis(self, index):
		batches, sources_num, sep_sources_num, _ = self.sample_mixture(index, "normal", merge=True)
		wav, note, onset = stack(batches)
		note[note > 1] = 1
		onset = onset.sum(0)
		onset[onset > 1] = 1
		mixture = wav.sum(0)
		sep = wav[0]
		batch = (mixture, sep, pad(note, 0), onset)
		return batch

		
	def clustering(self, index):
		target_instr = self.rng.randint(0, len(self._instr_tag))
		batches, sources_num, _, tags = self.sample_mixture(index, "pair", included=[target_instr])
		nonsil_batches, _, _, nonsil_tags = self.sample_mixture(index, "nonsil", included=[target_instr], excluded=[target_instr], all_source=True)

		wav, _, note = stack(batches)
		nonsil_wav, nonsil_note = stack(nonsil_batches)
		mixture = wav.sum(0)
		nonsil_mixture = nonsil_wav.sum(0)

		instr_mask = np.zeros([MAX_SOURCES_NUM, MAX_SOURCES_NUM])
		for i, ind in enumerate(tags):
			for j, nind in enumerate(nonsil_tags):
				if nind == ind:
					instr_mask[i, j] = 1

		batch = (mixture, pad(note, NOTES_NUM - 1), nonsil_mixture, nonsil_note, instr_mask)
		return batch


	def __len__(self):
		return SAMPLES_NUM
		#return self.samples_num

	def get_len(self):
		return self.__len__()

