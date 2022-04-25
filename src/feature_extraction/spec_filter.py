import librosa
import numpy as np
import copy

from utils.utilities import freq_2_note, note_2_freq, freq_2_bin

def init_freq_bin_mask(sample_rate, n_fft, notes_num, n_hamonic):
	freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
	SIL = notes_num - 1
	NB = 3

	max_n_hamonics = np.zeros([notes_num]) + n_hamonic
	notes_freq_range = np.zeros([notes_num, 6]) - 1
	for note in range(0, notes_num - 1):
		f0 = note_2_freq(note)
		f0_low = note_2_freq(note - NB)
		f0_up = note_2_freq(note + NB)
		
		low_bin = freq_2_bin(f0_low, freqs)
		up_bin = freq_2_bin(f0_up, freqs)


		notes_freq_range[note, 0] = f0_low
		notes_freq_range[note, 1] = f0_up

		notes_freq_range[note, 2] = low_bin
		notes_freq_range[note, 3] = up_bin if up_bin < len(freqs) else len(freqs) - 1

		assert low_bin >=0

		f0_low = note_2_freq(note - 0.5)
		f0_up = note_2_freq(note + 0.5)
		low_bin = freq_2_bin(f0_low, freqs)
		up_bin = freq_2_bin(f0_up, freqs)
		
		notes_freq_range[note, 4] = low_bin
		notes_freq_range[note, 5] = up_bin if up_bin < len(freqs) else len(freqs) - 1

		assert low_bin >=0

	note_hamonic = np.zeros([notes_num, n_hamonic + 1]) - 1
	for note in range(notes_num - 1):
		f0 = note_2_freq(note)
		for ham in range(n_hamonic + 1):
			fn = f0 * (ham + 1)
			n_note = int(freq_2_note(fn))
			if n_note < notes_num - 1 and n_note >= 0:
				note_hamonic[note, ham] = n_note

	
	#extended_notes_freq_range = copy.deepcopy(notes_freq_range)
	#for i in range(notes_num - 1):
	#	extended_notes_freq_range[i, 0, 2] = 0
	#	for ham in range(1, n_hamonic + 1):
	#		if extended_notes_freq_range[i, ham, 2] > extended_notes_freq_range[i, ham - 1, 3] + 1:
	#			extended_notes_freq_range[i, ham, 2] = (extended_notes_freq_range[i, ham, 2] + extended_notes_freq_range[i, ham - 1, 3] + 1) // 2
	#			extended_notes_freq_range[i, ham - 1, 3] = extended_notes_freq_range[i, ham, 2] - 1

	return notes_freq_range, note_hamonic, max_n_hamonics


def encode_f0(f0, mid_freq, low_freq, up_freq):
	if f0 < mid_freq:
		w = (f0 - low_freq) / (mid_freq - low_freq) * 0.5
	else:
		w = (up_freq - f0) / (up_freq - mid_freq) * 0.5 + 0.5
	return w


def spec_filter(spec, f0s_array, n_hamonic, notes_num, note_hamonic, notes_freq_range, pitch_shifting=0.):
	T = spec.shape[-1]
	SIL = notes_num - 1
	out_spec = np.zeros([n_hamonic + 1] + list(spec.shape))
	out_f0_w = np.zeros([T])
	out_note = np.zeros([n_hamonic + 1, T], dtype=np.int16) + SIL
	crop_filter = np.zeros([T], dtype=np.int16)
	for i, f0 in enumerate(f0s_array):
		max_bin = -1
		if f0 > 0:
			note = int(freq_2_note(f0)) + pitch_shifting
			if note > SIL or note < 0:
				continue
			freq = note_2_freq(note)
			out_note[:, i] = note_hamonic[note]
			for j in range(n_hamonic + 1):
				n_note = int(note_hamonic[note, j])
				if n_note < notes_num - 1:
					low_freq_bin = int(notes_freq_range[n_note, 2])
					up_freq_bin = int(notes_freq_range[n_note, 3]) + 1
					max_bin = up_freq_bin if up_freq_bin > max_bin else max_bin
					if low_freq_bin == -1:
						continue
					out_spec[j, low_freq_bin : up_freq_bin, i] = spec[low_freq_bin : up_freq_bin, i]
			out_f0_w[i] = encode_f0(f0, freq, notes_freq_range[note, 0], notes_freq_range[note, 1])
		crop_filter[i] = max_bin
	return out_spec, out_note, out_f0_w, crop_filter
