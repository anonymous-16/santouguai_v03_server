import numpy as np
import h5py
import os
import sys
import copy

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utilities import read_lst
from conf.feature import *

def get_max_min(x):
	t = copy.deepcopy(x)
	t[t == 0] = 2333
	return np.min(t, -1), np.max(x, -1)

def get_note_context(path):
	with h5py.File(path, 'r') as hf:
		f0_w = hf["f0_w"][:1]
		energy_w = hf["energy_w"][:]
	note_context = np.concatenate([f0_w, energy_w[None, :]], 0)
	min_c, max_c = get_max_min(note_context)
	return min_c, max_c

def min_val(x, y):
	return np.min(np.stack([x, y], 0), 0)

def max_val(x, y):
	return np.max(np.stack([x, y], 0), 0)



def get_max_min_context(file_lst_paths):
	file_lst = []
	for file_lst_path in file_lst_paths:
		file_lst += read_lst(file_lst_path)
	min_c = np.zeros([2]) + 23333
	max_c = np.zeros([2])
	for path in file_lst:
		min_context, max_context = get_note_context(path)
		min_c = min_val(min_c, min_context)
		max_c = max_val(max_c, max_context)
	#min_c[4] = 0
	
	return min_c, max_c


if __name__=="__main__":
	path = ["data-t/URMP_4_synthesis-2/Violin/train.lst"]
	print(get_max_min_context(path))
