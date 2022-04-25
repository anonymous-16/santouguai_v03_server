import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import h5py
from prefetch_generator import BackgroundGenerator

from utils.utilities import (note_2_freq, read_lst, one_hot)
#from feature_extraction.utilities import extract_context

from .get_conv_filters import get_filters

from conf.feature import *
from conf.sample import *
from conf.urmp import *


class UrmpDataset():
	def __init__(self, rng):
		self._file_lst = read_lst(TRAINING_LST_PATH)
		audios_num = len(self._file_lst)
		self._data = [None] * audios_num
		self._tracks_id = np.arange(audios_num)
		self._audios_num = audios_num
		self._rng = rng
		self._instr_2_track = {}

	def __get_next_track_id__(self):
		audios_num = self._audios_num
		self._rng.shuffle(self._tracks_id)
		nid = self._tracks_id[0]
		return nid

	def get_file_path(self):
		return self._path

	def next(self, pos, mode="normal"):
		rng = self._rng

		def load_file(track_id):
			if self._data[track_id] is None:
				hdf5_path = self._file_lst[track_id]
				data = []
				with h5py.File(hdf5_path, 'r') as hf:
					for k in hf.keys():
						data_unit = {}
						gr = hf[k]
						for v in gr.keys():
							if str.startswith(v, "instr"):
								instr = gr[v][:].astype(np.uint8).tostring().decode('ascii')
								data_unit[v] = instr
								if instr not in self._instr_2_track:
									self._instr_2_track[instr] = []
								if track_id not in self._instr_2_track[instr]:
									self._instr_2_track[instr].append(track_id)
							else:
								data_unit[v] = gr[v][:]
						data.append(data_unit)
				self._data[track_id] = data
			return self._data[track_id]

		def sample(spec_len, tracks_num, keep_instr=None, instr_lst=None):
			semi_tone = rng.randint(0, SHIFT_PITCH* 2 + 1) - SHIFT_PITCH
			st = rng.randint(0, spec_len - FRAME_DURATION * 2)
			ed = st + FRAME_DURATION
			wav_st = int(st * SAMPLE_RATE / FRAMES_PER_SEC)
			wav_ed = wav_st + int(DURATION * SAMPLE_RATE)
			tracks = np.arange(tracks_num)
			rng.shuffle(tracks)
			
			sub_tracks_num = rng.randint(1, tracks_num + 1)
			sub_tracks = list(tracks[:sub_tracks_num])
			if keep_instr is not None:
				for i in tracks:
					if instr_lst[i] == keep_instr and i not in sub_tracks:
						sub_tracks = [i] + sub_tracks[:-1]
			return semi_tone, st, ed, wav_st, wav_ed, sub_tracks

		def load_next(track_id):
			hf = load_file(track_id)
			tracks_num = len(hf)
			spec_len = hf[0]["note"].shape[-1]
			semi_tone, st, ed, wav_st, wav_ed, sub_tracks = sample(spec_len, tracks_num)	
			
			wavs = []
			notes = []
			onsets = []
			instrs = []
			for trk in sub_tracks:
				wavs.append(hf[trk][f"wav_{semi_tone}"][wav_st : wav_ed])
				n = hf[trk]["note"][st : ed]
				sil = n == 0
				n = n + semi_tone
				n[sil] = 0
				n[n > NOTES_NUM - 1] = NOTES_NUM - 1
				notes.append(n)
				onsets.append(hf[trk]["onset"][st : ed])
				instrs.append(hf[trk]["instr"])
			wav = np.stack(wavs, 0)
			note = one_hot(np.stack(notes, 0), NOTES_NUM).transpose(0, 2, 1)[:, 1:]
			onset = np.stack(onsets, 0)[:, None] * note
			return wav, note, onset, instrs

		def load_nonsil(instr):
			tracks = []
			while len(tracks) == 0:
				tracks = self._instr_2_track[instr]
				if len(tracks) == 0:
					load_file(self.__get_next_track_id__())
			rng.shuffle(tracks)
			track_id = tracks[0]
			hf = load_file(track_id)
			tracks_num = len(hf)
			spec_len = hf[0]["nonsil_note"].shape[-1]
			semi_tone, st, ed, wav_st, wav_ed, sub_tracks = sample(spec_len, tracks_num, keep_instr=instr, instr_lst=[v["instr"] for v in hf])

			wavs = []
			notes = []
			instrs = []
			for trk in sub_tracks:
				spec_len = hf[trk]["nonsil_note"].shape[-1]
				_, st, ed, wav_st, wav_ed, _ = sample(spec_len, tracks_num)
				wavs.append(hf[trk][f"nonsil_wav_{semi_tone}"][wav_st : wav_ed])
				n = hf[trk]["nonsil_note"][st : ed]
				sil = n == 0
				n = n + semi_tone
				n[sil] = 0
				n[n > NOTES_NUM - 1] = NOTES_NUM - 1
				notes.append(n)
				instrs.append(hf[trk]["instr"])
			wav = np.stack(wavs, 0)
			note = one_hot(np.stack(notes, 0), NOTES_NUM).transpose(0, 2, 1)[:, 1:]
			return wav, note, instrs

			return nonsil_wav, nonsil_note

		if mode == "normal":
			nid = self.__get_next_track_id__()
			track = load_next(nid)
		elif mode == "pair":
			track = load_next_pair(index)
		elif mode == "nonsil":
			track = load_nonsil(pos)
		return track

	def get_samples_num(self):
		return len(self._file_lst) * int(SAMPLE_RATE / SAMPLE_RES * 30 * 2)
		
