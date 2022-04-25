import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
import copy
plt.switch_backend('agg')

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utilities import mkdir
from feature_extraction.spec_filter import spec_filter, init_freq_bin_mask
from conf.feature import *


def draw_note_band():
	notes_freq_range, note_hamonic, bin_mask = init_freq_bin_mask(sample_rate=SAMPLE_RATE,
																			n_fft=N_FFT,
																			n_hamonic=N_HAMONIC,
																			notes_num=NOTES_NUM,
																			freq_bound_ratio=FREQ_BOUND_RATIO)

	
	freq_bin = bin_mask[-1].shape[-1]
	res = 40
	begin_note = 39
	spec = np.zeros([(NOTES_NUM - 1) * res, freq_bin])
	for i in range(NOTES_NUM - 1):
		spec[int(i * res) : int((i + 1) * res), int(notes_freq_range[i, 2]) : int(notes_freq_range[i, 3] + 1)] = (i % 2) + 1
	spec = spec[int(begin_note * res) :]
	plt.figure(figsize=(30, 60))
	plt.imshow(spec.transpose([1, 0]), cmap='GnBu', origin="lower")
	folder = "imgs"
	mkdir(folder)
	plt.xticks(np.arange(0, (NOTES_NUM - 1 - begin_note) * res,res), np.arange(begin_note, NOTES_NUM - 1, 1))
	plt.yticks(np.arange(0,freq_bin,1))
	plt.savefig(os.path.join(folder, "note_band.png"), bbox_inches='tight')


if __name__=="__main__":
	draw_note_band()
