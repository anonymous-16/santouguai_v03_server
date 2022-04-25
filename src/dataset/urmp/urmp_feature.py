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

from utils.utilities import (freq_2_note, mkdir, get_filename, get_process_groups, read_lst, write_lst)

from conf.feature import *
from conf.urmp import *

et = 1e-8

def sec2frame(s):
	return round(s * FRAMES_PER_SEC)

def get_frame_num(ed, st):
	spec_len = int((ed -st) * FRAMES_PER_SEC)
	y_len = int(spec_len / FRAMES_PER_SEC * SAMPLE_RATE)
	while not int(1000 * y_len / SAMPLE_RATE) == int(1000 * spec_len / FRAMES_PER_SEC):
		spec_len += 1
		y_len = int(spec_len / FRAMES_PER_SEC * SAMPLE_RATE)
	return spec_len, y_len


def read_note_onset(path, wav_len):
	labels = read_lst(path, "\t\t")
	spec_len = int((wav_len / SAMPLE_RATE + 1) * FRAMES_PER_SEC)
	notes = np.zeros([spec_len])
	onsets = np.zeros([spec_len])
	nonsil_spec_mask = np.zeros([spec_len])
	nonsil_mask = np.zeros([wav_len])	
	for line in labels:
		ost = float(line[0])
		oed = float(line[2]) + ost
		
		st = sec2frame(ost)
		ed = sec2frame(oed)

		onset_ed = st + ONSET_RES if st + ONSET_RES < ed else ed
		n = freq_2_note(float(line[1]))
		notes[st : ed] = int(n)
		onsets[st : onset_ed] = 1
		
		spec_len, y_len = get_frame_num(oed, ost)
		nonsil_spec_mask[st : st + spec_len] = 1
		wav_st = round(ost * SAMPLE_RATE)
		nonsil_mask[wav_st : wav_st + y_len] = 1
	print(np.max(notes), np.min(notes))
	notes[notes > NOTES_NUM - 1] = NOTES_NUM - 1
	notes[notes < 0 ] = 0
	return notes, onsets, nonsil_spec_mask, nonsil_mask
		

def extract_spec(data):
	output_path = data["output_path"]
	data = data["data"]
	print(output_path)
	with h5py.File(output_path, 'w') as hf:
		for i, data_unit in enumerate(data):
			instr, wav_path, f0_path, note_path= data_unit	
			instr = np.frombuffer(instr.encode(), "u1").astype('float64')
			ori_wav, sr = librosa.load(wav_path)
			re_wav = librosa.resample(ori_wav, sr, SAMPLE_RATE)
			labels, onsets, nonsil_spec_mask, nonsil_mask = read_note_onset(note_path, re_wav.shape[-1])	
			nonsil_labels = labels[nonsil_spec_mask == 1]	
			gr = hf.create_group(f"track_{i + 1}")
			gr.create_dataset(name="instr", data=instr)
			gr.create_dataset(name="note", data=labels, dtype=np.int16)
			gr.create_dataset(name="onset", data=onsets, dtype=np.int16)
			gr.create_dataset(name="nonsil_note", data=nonsil_labels, dtype=np.int16)

			for j in range(SHIFT_PITCH * 2 + 1):
				semi = j - SHIFT_PITCH
				if semi == 0:
					wav = re_wav
				else:
					wav =	librosa.effects.pitch_shift(ori_wav, sr=sr, n_steps=semi)
					wav = librosa.resample(wav, sr, SAMPLE_RATE)
				nonsil_wav = wav[nonsil_mask == 1]
				gr.create_dataset(name=f"wav_{semi}", data=wav, dtype=np.float32)
				gr.create_dataset(name=f"nonsil_wav_{semi}", data=nonsil_wav, dtype=np.float32)


def pack_urmp_dataset_to_hdf5(args):

	dataset_dir = args.dataset_dir
	feature_dir = args.feature_dir
	process_num = args.process_num

	audios_num = 0
	
	audio = []
	mkdir(feature_dir)

	for folder in os.listdir(dataset_dir):
		if str.startswith(folder, "._"):
			continue
		meta_data = folder.split('_')
		if len(meta_data) < 4:
			continue
		data = []
		id = meta_data[0]
		name = meta_data[1]
		sources = meta_data[2:]
		output_path = os.path.join(feature_dir, f"{id}_{name}.h5")
		data = {"data" : [], "output_path" : output_path}
		for j, s in enumerate(sources):
			instr = INSTRUMENTS_MAP[s]
			source_name = f"{name}_{id}_{j + 1}"
			instr_folder = os.path.join(feature_dir, instr)
			song_folder = os.path.join(instr_folder, source_name)

			data["data"].append([instr, os.path.join(dataset_dir, folder, f'AuSep_{j + 1}_{s}_{id}_{name}.wav'),
																				os.path.join(dataset_dir, folder, f'F0s_{j + 1}_{s}_{id}_{name}.txt'),
																				os.path.join(dataset_dir, folder, f'Notes_{j + 1}_{s}_{id}_{name}.txt'),])
		audio.append(data)

	feature_time = time.time()

	audios_num = len(audio)
	print(f"The total number of the mixture audio is {audios_num}")

	def process_group(st, ed, total_num, pid):
		print(f"process {pid + 1} starts")
		for audio_unit in audio[st : ed]:
			extract_spec(audio_unit)
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

	args = parser.parse_args()
	pack_urmp_dataset_to_hdf5(args)
		
