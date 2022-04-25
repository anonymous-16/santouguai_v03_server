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
		return np.pad(x, ((0, pad_len), (0, 0)), 'constant', constant_values=(pad_value, pad_value))
	return x

def stack(x, axis=0):
	output = []
	for line in x:
		if len(output) == 0:
			for i in range(len(line)):
				output.append([])
		for i, d in enumerate(line):
			output[i].append(d)
	output = [np.stack(o, axis) for o in output]
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
				instrs.append(r)
		return instrs

	def sample_mixture(self, index, mode, repeat=True, all_source=False, included=[], excluded=[]):
		sources_num = self.rng.randint(1, MAX_SOURCES_NUM + 1) if not all_source else MAX_SOURCES_NUM
		mixture = self.sample_unit(sources_num, included=included, excluded=excluded, repeat=repeat)

		batches = []
		for i, instr_id in enumerate(mixture):
			tag = self._instr_tag[instr_id]
			batch = self.dataset[tag].next(index, mode=mode)
			batches.append(batch)
		return batches, sources_num

	def disentangle(self, index):
		batches, sources_num = self.sample_mixture(index, "normal")
		wav_a, wav_b, note_a, f0_n_a, f0_a, en_a = stack(batches)
		mix = wav_a.sum(0)
		shifted_mix = wav_b.sum(0)
		batch = (mix, shifted_mix, wav_a[0], wav_b[0],\
								note_a[0], f0_n_a[0], f0_a[0], en_a[0])
		return batch

	def separate(self, index):
		batches, sources_num = self.sample_mixture(index, "pair")
		wav, f0, note = stack(batches)
		sep_sources_num = self.rng.randint(1, sources_num + 1)
		mixture = wav.sum(0)
		sep = wav[:sep_sources_num].sum(0)
		batch = (mixture, sep, pad(f0, F0S_NUM))
		return batch

	def clustering(self, index):
		target_instr = self.rng.randint(0, len(self._instr_tag)) 
		batches, sources_num = self.sample_mixture(index, "pair", included=[target_instr], excluded=[target_instr])
		nonsil_batches, _ = self.sample_mixture(index, "nonsil", included=[target_instr], excluded=[target_instr], all_source=True)
		
		wav, f0, _ = stack(batches)
		nonsil_wav, nonsil_f0 = stack(nonsil_batches)
		mixture = wav.sum(0)
		nonsil_mixture = nonsil_wav.sum(0)
		batch = (mixture, f0[0], nonsil_mixture, nonsil_f0)
		return batch


	def __len__(self):
		return SAMPLES_NUM
		#return self.samples_num

	def get_len(self):
		return self.__len__()

