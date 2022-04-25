import os
import time
import numpy as np
import json
import pretty_midi
from conf.feature import *

et = 1e-8

def trim_silence(audio, sample_rate, min_silence_duration=0.3):
	tfm = sox.Transformer()
	tfm.silence(min_silence_duration=min_silence_duration,
							buffer_around_silence=False)
	audio = tfm.build_array(input_array=np.expand_dims(audio, axis=1),
													sample_rate_in=sample_rate)
	return audio

def numpy_2_midi(tracks, output_path, notes_num=NOTES_NUM, begin_note=0, fs=FRAMES_PER_SEC * 1.):
	music = pretty_midi.PrettyMIDI()
	
	for notes in tracks:
		piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
		piano = pretty_midi.Instrument(program=piano_program)
		for i in range(notes.shape[0]):
			st = -1
			for j in range(notes.shape[1] + 1):
				if j == notes.shape[1] or notes[i, j] == 0:
					if st > -1:
						note = pretty_midi.Note(velocity=100, pitch=int(i + begin_note), start=st / fs, end=j / fs)
						piano.notes.append(note)
					st = -1
				elif notes[i, j] == 1:
					if st == -1:
						st = j
		music.instruments.append(piano)
	music.write(output_path)

def load_json(path):
	with open(path,'r') as load_f:
		load_dict = json.load(load_f)
	return load_dict

def save_json(path, data):
	with open(path,'w') as f:
		json.dump(data,f) 
	
def print_dict(x):
	for key in x:
		print(key, x[key])


def compute_time(event, pre_time):
	cur_time = time.time()
	print(f'{event} use', cur_time - pre_time)
	return cur_time

def encode_mu_law(x, mu=256):
	mu = mu - 1
	fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
	return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def decode_mu_law(y, mu=256):
	mu = mu - 1
	fx = (y - 0.5) / mu * 2 - 1
	x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
	return x


def read_config(config_path, name):
	config = configparser.ConfigParser()
	config.read(config_path)
	return config[name]


def dict2str(dic, pre):
	res = ''
	for i, d in enumerate(dic):
		if i == 0:
			res += pre
		res += d + ' :'
		val = dic[d]
		if type(val) is dict:
			res += '\n' + dict2str(val, pre + '\t') + '\n'
		else:
			res += f'\t{val}\t'

	return res		

def save_score(path, score):
	mkdir(path, is_file=True)
	res = dict2str(score, '')
	write_lst(path, [res])
	return res
		
def get_process_groups(audio_num, process_num):
	assert audio_num > 0 and process_num > 0
	if process_num > audio_num:
		process_num = audio_num
	audio_num_per_process = (audio_num + process_num - 1) // process_num

	reduce_id = process_num - (audio_num_per_process * process_num - audio_num)

	groups = []
	cur = 0
	for i in range(process_num):
		if i == reduce_id:
			audio_num_per_process -= 1
		groups += [[cur, cur + audio_num_per_process]]
		cur += audio_num_per_process
	return groups


def mkdir(fd, is_file=False):
	fd = fd.split('/')
	fd = fd[:-1] if is_file else fd
	ds = []
	for d in fd:
		ds.append(d)
		d = "/".join(ds)
		if not d == "" and not os.path.exists(d):
			os.makedirs(d)
		
		
def get_filename(path):
	path = os.path.realpath(path)
	na_ext = path.split('/')[-1]
	na = os.path.splitext(na_ext)[0]
	return na


def traverse_folder(folder):
	paths = []
	names = []
	
	for root, dirs, files in os.walk(folder):
		for name in files:
			filepath = os.path.join(root, name)
			names.append(name)
			paths.append(filepath)
			
	return names, paths


def note_to_freq(piano_note):
	return 2 ** ((piano_note - 39) / 12) * 440
	

def float32_to_int16(x):
	x = np.clip(x, -1, 1)
	assert np.max(np.abs(x)) <= 1.
	return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
	return (x / 32767.).astype(np.float32)
	

def pad_truncate_sequence(x, max_len):
	if len(x) < max_len:
		return np.concatenate((x, np.zeros(max_len - len(x))))
	else:
		return x[0 : max_len]

def read_lst(lst_path, split=None):
	with open(lst_path) as f:
		data = f.readlines()
	data = [d.rstrip() for d in data]
	if split is not None:
		data = [d.split(split) for d in data]
	return data

def write_lst(lst_path, lst):
	lst = [str(l) for l in lst]
	with open(lst_path, 'w') as f:
		f.writelines('\n'.join(lst))

	
def parse_frameroll2annotation(frame_roll, frames_per_second=100, notes_num=88):
	pre = notes_num
	st = -1
	est = []
	preds = np.pad(frame_roll,(0,1), 'constant', constant_values=(0, notes_num))
	for i in range(frame_roll.shape[0]):
		if not frame_roll[i] == pre:
			if st > -1 and not pre == notes_num:
				est.append(\
					'%f\t%f\t%d' % (st * 1.0 / frames_per_second, i * 1.0 / frames_per_second, pre))
			st = i
		pre = frame_roll[i]
	return est


def one_hot(x, classes_num):
	x = x.astype(np.int16)
	x_eye = np.eye(classes_num)
	return x_eye[x]

def align(x, y):
	if x.shape[-1] < y.shape[-1]:
		x = np.pad(x, (0, y.shape[-1] - x.shape[-1]), 'constant', constant_values=(0, 0))
	elif x.shape[-1] > y.shape[-1]:
		x = x[:y.shape[-1]]
	return x

def freq_2_bin(freq, freqs, ham=0):
	freq = freq * (ham + 1)
	if freq >= freqs[-1]:
			return len(freqs) - 1
	for i in range(len(freqs)):
		if freq < freqs[i]:
			if i > 0 and freq - freqs[i - 1] < freqs[i] - freq:
				return i - 1
			return i

def mkdir(fd, is_file=False):
	fd = fd.split('/')
	fd = fd[:-1] if is_file else fd
	ds = []
	for d in fd:
		ds.append(d)
		d = "/".join(ds)
		if not d == "" and not os.path.exists(d):
			os.makedirs(d)

def freq_2_note(freq):
	freq = float(freq)
	note = round(12 * np.log2(freq / 440)) + 69
	assert note > 0 and note < NOTES_NUM
	return note

def freq_dis(freq1, freq2, res):
	dis = res * 12 * np.log2(freq1 / 440) - res * 12 * np.log2(freq2 / 440)
	sign = dis
	dis = np.abs(dis)

	if dis > res / 2:
		dis = res / 2 + (dis - res / 2) / res

	n = int(dis * 4) // res
	remain = dis - n * res / 4

	q = 1. / 2.
	dis = res / 4 * (1 - q**n) / (1 - q) + remain * (q**n)

	
	if sign < 0:
		dis = -dis

	if dis == res / 2:
		print(n, remain, dis)

	return dis

def note_2_freq(note):
	note = float(note)
	freq = (2**((note - 69) / 12)) * 440
	return freq

def freq_2_vq(freq, note):
	if note == NOTES_NUM - 1:
		return F0S_NUM, F0S_NUM, 0

	f0_n = F0_RES * 12 * np.log2(freq / 440) + 69 * F0_RES
	if f0_n > F0S_NUM:
		f0_n = F0S_NUM
	if f0_n < 0:
		f0_n = 0
		
	res = F0_RES
	n_freq = note_2_freq(note)
	dis = freq_dis(freq, n_freq, res )

	dis += (res / 2)
	
	w = dis / res
	dis = (note + w) * res
	if dis < 0:
		dis = 0

	f0_vq = dis

	return f0_vq, f0_n, freq
