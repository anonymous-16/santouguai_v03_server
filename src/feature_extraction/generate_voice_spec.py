import sys
import os
import numpy as np
import librosa
import scipy
from scipy.io import wavfile
import h5py
import math
import crepe


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utilities import trim_silence, read_lst, freq_2_note, note_2_freq, freq_2_bin
from feature_extraction.spec_filter import spec_filter, init_freq_bin_mask
from utils.utilities import mkdir
from conf.feature import *

mel_basis = librosa.filters.mel(SAMPLE_RATE, N_FFT, n_mels=N_MELS)


def read_f0(path, frames_per_sec):
	lines = read_lst(path, split="\t")
	f0s = []
	for line in lines:
		t = int(float(line[0]) * frames_per_sec)
		f0 = float(line[1])
		f0s.append([t, f0])
	f0s_array = np.zeros([f0s[-1][0] + 1])
	for f0 in f0s:
		f0s_array[f0[0]] = f0[1]
	return f0s_array


def read_onset_note(path, frames_per_sec, spec_len, semi_tone):
	lines = read_lst(path, split="\t\t")
	onsets_array = np.zeros([spec_len])
	notes_array = np.zeros([3, spec_len], dtype=np.int16)
	notes_array[0, :] = NOTES_NUM - 1
	notes_array[1:, :] = -1

	for line in lines:
		t = int(float(line[0]) * frames_per_sec)
		ed = int(float(line[0]) * frames_per_sec + float(line[2]) * frames_per_sec) + 1
		note = int(freq_2_note(float(line[1])))
		note = note + semi_tone
		yinqu = (note + 9) // 12
		octa = (note + 9) % 12

		onsets_array[t] = 1
		notes_array[0, t : ed] = note
		notes_array[1, t : ed] = yinqu
		notes_array[2, t : ed] = octa
		
	return onsets_array, notes_array

def load_audio(wav_path):
	hq_y, _ = librosa.load(wav_path, sr=44100, mono=True)
	y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
	return y, hq_y

def wav_2_spec(hq_y, y, semi_tone=0):
	if not semi_tone == 0:
		y = librosa.effects.pitch_shift(hq_y, sr=44100, n_steps=semi_tone)
		y = librosa.resample(y, 44100, SAMPLE_RATE)

	spec_angle = librosa.stft(y,
													n_fft=N_FFT,
													win_length=WINDOW_SIZE,
													hop_length=HOP_SIZE,
													window=scipy.signal.windows.hann,
													pad_mode=PAD_MODE)
	spec = np.abs(spec_angle)
	return spec, y

def spec_2_wav(spec, angle, y_len):
	spec_angle = spec * (np.cos(angle) + np.sin(angle)*1j)
	y = librosa.istft(spec_angle,
												win_length=WINDOW_SIZE,
												hop_length=HOP_SIZE,
												window=scipy.signal.windows.hann,
												length=y_len)
	return y


def save_audio(path, audio):
	wavfile.write(path, SAMPLE_RATE, audio[:, None])



def predict_f0(signal, sr, length):
	f0 = crepe.predict(
		signal,
		sr,
		step_size=int(HOP_SIZE / sr * 1000),
		verbose=1,
		center=True,
		viterbi=True,
	)
	f0 = f0[1].reshape(-1)[:-1]
	print(f0.shape, signal.shape, length)
	if f0.shape[-1] != length:
		f0 = np.interp(
			np.linspace(0, 1, length, endpoint=False),
			np.linspace(0, 1, f0.shape[-1], endpoint=False),
			f0,
		)
	print(f0.shape)
	return f0


def f0_2_w(f0s_array, notes_array):
	w_array = np.zeros([1, f0s_array.shape[-1]])
	for i, f0 in enumerate(f0s_array):
		if notes_array[i] < NOTES_NUM - 1:
			note = notes_array[i]
			freq = round(note_2_freq(note), 1)
			w_array[0, i] = f0s_array[i] / freq
			w_array[0, i] = (1 -	w_array[0, i] * 0.5) if w_array[0, i] <=1 else freq / f0s_array[i] * 0.5
	return w_array


def extract_query(spec, note):
	index = note < NOTES_NUM - 1
	query = spec[:, index]
	return query


def extract_en(spec):
	en = spec.sum(0)
	ave = en.mean()
	return en / ave

def extract_spec(data_path_lst, save_hamonic=True):
	
	for song_folder, wav_path, f0_path, note_path in data_path_lst:
		hdf5_path = os.path.join(song_folder,	"data.h5")
		print(hdf5_path)
		if not hdf5_path == "dataset/hdf5s/URMP_raw/Viola/Rondeau_35_3/data.h5":
			continue
	
		mkdir(song_folder)

		with h5py.File(hdf5_path, 'w') as hf:	
	
			y, hq_y = load_audio(wav_path)
			q_y = trim_silence(y, SAMPLE_RATE)
			q_hq_y = trim_silence(hq_y, 44100)

			for i in range(SHIFT_PITCH * 2 + 1):
				semi_tone = i - SHIFT_PITCH		
				spec, yt = wav_2_spec(hq_y, y, semi_tone=semi_tone)		
				spec_q, yt_q = wav_2_spec(q_hq_y, q_y, semi_tone=semi_tone)			
				#query_mel_spec = np.dot(mel_basis, query_spec)
	
				f0 = predict_f0(yt, SAMPLE_RATE, spec.shape[-1])	
				f0_q = predict_f0(yt_q, SAMPLE_RATE, spec_q.shape[-1])

				onsets_array, notes_array = read_onset_note(note_path, FRAMES_PER_SEC, f0.shape[-1], semi_tone)

	
				#hf.create_dataset(name=f"spec_pitch-shifting_{semi_tone}", data=spec, dtype=np.float)
				hf.create_dataset(name=f"f0_pitch-shifting_{semi_tone}", data=f0, dtype=np.float)
				hf.create_dataset(name=f"f0_nonsil_pitch-shifting_{semi_tone}", data=f0_q, dtype=np.float)
				hf.create_dataset(name=f"note_pitch-shifting_{semi_tone}", data=notes_array[0].astype(np.int16), dtype=np.int16)
				#hf.create_dataset(name=f"query_mel_spec_pitch-shifting_{semi_tone}", data=query_mel_spec, dtype=np.float)
				hf.create_dataset(name=f"y_pitch-shifting_{semi_tone}", data=yt, dtype=np.float)
				hf.create_dataset(name=f"y_nonsil_pitch-shifting_{semi_tone}", data=yt_q, dtype=np.float)


if __name__=="__main__":
	path_lst = {
		"Jupiter" : {
			"f0" : "data/annotation/F0s_1_vn_01_Jupiter.txt",
			"wav" : "data/audio/AuSep_1_vn_01_Jupiter.wav"
		},
		"Sonata" : {
			"f0" : "data/annotation/F0s_1_vn_02_Sonata.txt", 
			"wav" : "data/audio/AuSep_1_vn_02_Sonata.wav" 
		},
		"Hark" : {
			"f0" : "data/annotation/F0s_1_vn_13_Hark.txt",
			"wav" : "data/audio/AuSep_1_vn_13_Hark.wav"
		}
	}
	output_folder = "data/output_audio"
	extract_spec(path_lst, output_folder)	
