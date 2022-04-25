import sys
import os
import numpy as np
import librosa
import scipy
from scipy.io import wavfile
import cmath

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utilities import *
from conf.feature import *
from conf.synthesizer import *

freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

BAND_WIDTH = 5
BOUND_RATIO = 0.05

def freq_2_bin(freq, ham=0):
	freq = freq * (ham + 1)
	if freq >= freqs[-1]:
			return len(freqs)
	for i in range(len(freqs)):
		if freq < freqs[i]:
			if freq - freqs[i - 1] < freqs[i] - freq:
				return i - 1
			return i

def init_note_freq_band_width(freq_bound_ratio=BOUND_RATIO, n_hamonic=N_HAMONIC, notes_num=NOTES_NUM, sample_rate=SAMPLE_RATE, freq_bin=N_FFT // 2 + 1):
	freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=freq_bin)
	freq_bin_mask = np.zeros([notes_num, n_hamonic + 1, freq_bin + 1])
	for i in range(notes_num):
		freq = note_2_freq(i)
		for j in range(n_hamonic + 1):
			nfreq = freq + j * freq
			note = freq_2_note(nfreq)
			up_bound_freq = note_2_freq(note + 1) - (note_2_freq(note + 1) - note_2_freq(note)) * (nfreq / sample_rate + freq_bound_ratio)
			low_bound_freq = note_2_freq(note - 1) + (note_2_freq(note) - note_2_freq(note - 1)) * (nfreq / sample_rate + freq_bound_ratio)
			up_freq_bin = freq_2_bin(up_bound_freq)
			low_freq_bin = freq_2_bin(low_bound_freq)
			freq_bin_mask[i, j, low_freq_bin : up_freq_bin + 1] = 1

	mask = 1 - freq_bin_mask[:, 0]
	for i in range(n_hamonic):
		freq_bin_mask[:, i + 1] = freq_bin_mask[:, i + 1] * mask
		mask = 1 - mask + freq_bin_mask[:, i + 1]
		mask[mask > 1] = 1
		mask = 1 - mask

	index = np.array(np.arange(freq_bin_mask.shape[-1]))
	freq_bound = np.zeros([notes_num, n_hamonic + 1, 2], dtype=np.int16)
	for i in range(notes_num):
		for j in range(n_hamonic + 1):
			index = np.array(np.arange(freq_bin_mask.shape[-1])) + 1
			tindex = index * freq_bin_mask[i, j]
			tindex = tindex[tindex > 0] - 1
			if tindex.shape[0] == 0:
				continue
			up_freq_bin = int(np.max(tindex))
			low_freq_bin = int(np.min(tindex))
			freq_bound[i, j, 0] = low_freq_bin
			freq_bound[i, j, 1] = up_freq_bin
	return freq_bin_mask, freq_bound


freq_bin_mask, freq_bound = init_note_freq_band_width()

def extract_fn(spec, f0s_array, ham):
	f0s_array = align(f0s_array, spec)
	spec = np.pad(spec, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
	tspec = np.zeros_like(spec)
	ham = int(ham)
	for i, f0 in enumerate(f0s_array):
		tspec[:, i] = 0
		if f0 > 0:
			note = int(freq_2_note(f0))
			low_freq_bin = freq_bound[note, ham, 0]
			up_freq_bin = freq_bound[note, ham, 1]
			tspec[low_freq_bin : up_freq_bin + 1, i] = spec[low_freq_bin : up_freq_bin + 1, i]
	return tspec[:-1]


def extract_spec(spec, f0s_array):
	n_hamonic = N_HAMONIC
	specs = []
	for i in range(n_hamonic + 1):
		spec_t, f0_w = extract_fn(spec, f0s_array, ham=i)
		specs.append(spec_t)
		f0_w.append()
	spec = np.stack(specs, 0)
	f0s = np.zeros_like(f0s_array) + 0.5
	for i, f0 in enumerate(f0s_array):
		if f0 > 0:
			note = int(freq_2_note(f0))
			low_freq = note_2_freq(note - 1)
			up_freq = note_2_freq(note + 1)
			freq = note_2_freq(note)
			if f0 < freq:
				f0 = (f0 - low_freq) / (freq - low_freq) * 0.5
			else:
				f0 = (up_freq - f0) / (up_freq - freq) * 0.5 + 0.5
			f0s[i] = f0
	return specs, f0s


def get_wav(wav_path, notes_array, semi_tone=0.):
	y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

	spec_angle = librosa.stft(y, 
														n_fft=N_FFT,
														win_length=WINDOW_SIZE,
														hop_length=HOP_SIZE,
														window=scipy.signal.windows.hann,
														pad_mode=PAD_MODE)
	spec = np.abs(spec_angle)
	angle = np.angle(spec_angle)

	spec_data, f0s = extract_spec(spec, notes_array)
	real = spec_data * np.cos(angle)
	imag = spec_data * np.sin(angle)
	
	voice_spec = spec_data[0]

	noise_spec = spec - voice_spec

	spec_angle = voice_spec * (np.cos(angle) + np.sin(angle)*1j)
	y_out = librosa.istft(spec_angle,
											 	win_length=WINDOW_SIZE,
												hop_length=HOP_SIZE,
												window=scipy.signal.windows.hann,
												length=len(y))
	spec_angle = noise_spec * (np.cos(angle) + np.sin(angle)*1j)
	y_noise = librosa.istft(spec_angle,
												win_length=WINDOW_SIZE,
												hop_length=HOP_SIZE,
												window=scipy.signal.windows.hann,
												length=len(y))


	return y[None, :], y_out[None, :], y_noise[None, :], np.stack([real, imag, spec_data], 1), f0s

def read_f0(path):
	with open(path, "r") as f:
		lines = f.readlines()
	lines = [l.rstrip().split("\t") for l in lines]
	f0s = []
	for line in lines:
		t = int(float(line[0]) * FRAMES_PER_SEC)
		f0 = float(line[1])
		f0s.append([t, f0])
	f0s_array = np.zeros([f0s[-1][0] + 1])
	for f0 in f0s:
		f0s_array[f0[0]] = f0[1]
	return f0s_array


if __name__=="__main__":
	input_audio_dir = "data/audio"
	input_annotation_dir = "data/annotation"
	output_audio_dir ="data/output_audio"
	mkdir(output_audio_dir)
	for fname in os.listdir(input_audio_dir):
		if not str.endswith(fname, ".wav"):
			continue
		path = os.path.join(input_audio_dir, fname)
		annotation_path = str.replace(fname, "AuSep", "F0s")
		annotation_path = str.replace(annotation_path, ".wav", ".txt")
		annotation_path = os.path.join(input_annotation_dir, annotation_path)
		f0s_array = read_f0(annotation_path)
		ori_wav, wav, noise, spec, f0 = get_wav(path, f0s_array)
		path = os.path.join(output_audio_dir, f"spec_{fname}")
		np.save(path, spec)
		path = os.path.join(output_audio_dir, f"f0_{fname}")
		np.save(path, f0)
		path = os.path.join(output_audio_dir, f"{fname}")
		wavfile.write(path, SAMPLE_RATE, np.transpose(wav, [1, 0]))
		path = os.path.join(output_audio_dir, f"noise_" + fname)
		wavfile.write(path, SAMPLE_RATE, np.transpose(noise, [1, 0]))
		path = os.path.join(output_audio_dir, f"ori_" + fname)
		wavfile.write(path, SAMPLE_RATE, np.transpose(ori_wav, [1, 0]))

