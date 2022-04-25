import numpy as np
import argparse
import csv
import os
import time
import h5py
import librosa
import multiprocessing
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from utils.utilities import (mkdir, get_filename, get_process_groups, read_lst, write_lst)

from feature_extraction.generate_voice_spec import extract_spec
from conf.feature import *
from conf.urmp import *

et = 1e-8


def pack_urmp_dataset_to_hdf5(args):

	dataset_dir = args.dataset_dir
	feature_dir = args.feature_dir
	process_num = args.process_num
	instrument = args.instrument

	audios_num = 0
	
	audio = []

	for folder in os.listdir(dataset_dir):
		if str.startswith(folder, "._"):
			continue
		meta_data = folder.split('_')
		if len(meta_data) < 4:
			continue	
		id = meta_data[0]
		name = meta_data[1]
		sources = meta_data[2:]
		for j, s in enumerate(sources):
			instr = INSTRUMENTS_MAP[s]
			if not instr == instrument:
				continue
			source_name = f"{name}_{id}_{j + 1}"
			instr_folder = os.path.join(feature_dir, instr)
			song_folder = os.path.join(instr_folder, source_name)

			audio.append([song_folder, os.path.join(dataset_dir, folder, f'AuSep_{j + 1}_{s}_{id}_{name}.wav'),
																	os.path.join(dataset_dir, folder, f'F0s_{j + 1}_{s}_{id}_{name}.txt'),
																	os.path.join(dataset_dir, folder, f'Notes_{j + 1}_{s}_{id}_{name}.txt'),])


	feature_time = time.time()

	audios_num = len(audio)
	print(f"The total number of the mixture audio is {audios_num}")

	def process_group(st, ed, total_num, pid):
		print(f"process {pid + 1} starts")
		extract_spec(audio[st : ed], save_hamonic=False)
		print(f"process {pid + 1} ends")


	audio_groups = get_process_groups(audios_num, process_num)
	for pid, (st, ed) in enumerate(audio_groups):
		p = multiprocessing.Process(target = process_group, args = (st, ed, audios_num, pid))
		p.start()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
	parser.add_argument('--feature_dir', type=str, required=True, help='Directory to store generated files.')
	parser.add_argument('--process_num', type=int, required=True, help='Number of processes.')
	parser.add_argument('--instrument', type=str, required=True, help='Instrument.')

	args = parser.parse_args()
	pack_urmp_dataset_to_hdf5(args)
		
