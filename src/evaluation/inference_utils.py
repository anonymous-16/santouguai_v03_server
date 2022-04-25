import os
import torch
import torchaudio
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

from utils.utilities import mkdir, numpy_2_midi
from utils.torch_utilities import (predict, onehot_tensor, spec_2_wav, wav_2_spec)

from models.models import DisentanglementModel
from conf.feature import *
from conf.inference import *

#HAM_WIN = scipy.signal.windows.hamming(SMOOTH_WIN, sym=True)
#HAM_WIN = HAM_WIN / HAM_WIN.sum()

def device(x):
	return torch.from_numpy(x).float().cuda()

def num(x):
	return int(x.item())

def manul_divide(note, groups, n_groups):
	clusters = onehot_tensor(note, NOTES_NUM)[:, :-1].transpose(-1, -2)
	if clusters.shape[-1] < groups.shape[-1]:
		clusters = F.pad(clusters, (0, groups.shape[-1] - clusters.shape[-1]), "constant", 0)

	target = torch.zeros_like(clusters)
	for i in range(n_groups):
		ind = groups == i + 1
		samples_num = ind.sum()
		hit_num = clusters[ind].sum()
		if hit_num * 1. / samples_num > 0.2:
			target[ind] = 1
	return target


def group_note(note_target, onset_target, rep, index, n_clusters):
	c = n_clusters + 1
	MAXB = 23333
	T, NN = note_target.shape
	TB = 5

	index = index.long()
	note_res = torch.zeros_like(note_target) + MAXB

	for i in range(index.shape[0]):
		x, y = index[i]
		if x < TB:
			note_res[x, y] = c
			c += 1
			continue
	
		if note_target[x - TB:x, y].sum() > 0 and (onset_target[x, y] == 0):
			dis = ((rep[x - TB:x, y] - rep[x : x + 1, y])**2).sum(-1)
			dis[note_target[x - TB:x, y] == 0] = MAXB
			ind = torch.argmin(dis) + x - TB
			if note_res[ind, y] < MAXB:
				note_res[x, y] = note_res[ind, y]
			else:
				note_res[x, y] = c
				c += 1
		else:
			note_res[x, y] = c
			c += 1

	note_res[note_res == MAXB] = 0		
	#ind = ((1 - mono_mask) * note_target) == 1
	#hit = torch.unique(note_res[ind])
	#for ind in hit:
	#	note_res[note_res == ind] = - ind

	c -= 1
	#for i in range(c):
	#	ind = note_res == i + 1
	#	if ind.sum() < TB:
	#		note_res[ind] = 0

	return note_res, c

def group(note_target, onset_target, rep, n_clusters):

	rep = rep.transpose(-1, -2)	
	note_target, onset_target, rep = [v.transpose(0, 1) for v in [note_target, onset_target, rep]]
	x_ind = (torch.arange(note_target.shape[0])[:, None].to(note_target.device) + torch.zeros_like(note_target)).flatten()
	y_ind = (torch.arange(note_target.shape[1])[None, :].to(note_target.device) + torch.zeros_like(note_target)).flatten()
	
	fl_data = note_target.flatten()
	index = fl_data > 0
	index = torch.stack([x_ind[index], y_ind[index]], 1)
	print(f"note grouping {index.shape[0]} samples...")	
	group_res, c = group_note(note_target, onset_target, rep, index, n_clusters)
	print(f"note grouping done | {index.shape[0]} -> {c}")
	return group_res.transpose(0, 1).long(), c

def compute_center(abt_rep, f0, group_res, index, n_clusters):
	c = len(index)
	sil = index.sum()
	print(f"\t1. computing the center of {sil} groups...")
	c = len(index)
	f0, abt_rep = [v.flatten(0, 1) for v in [f0, abt_rep]]
	feature_dim = abt_rep.shape[-1]
	clustering_embedding = torch.zeros([c, feature_dim]).to(abt_rep.device)
	for i in range(c):
		if not index[i]:
			continue
		ind = group_res == i + 1
		if ind.sum() == 0:
			ind = group_res == - (i + 1)
			index[i] = False
		if ind.sum() == 0:
			continue
		v = abt_rep[ind].sum(0) / ind.sum()
		clustering_embedding[i] = v
	print(f"\t	 compute done | {sil} -> {index.sum()}")
	return clustering_embedding, group_res, index

def kmeans(group_res, clustering_embedding, index, n_clusters, init=None):
	c = len(index)
	print("\t2. kmeans...")
	y = clustering_embedding[index].cpu().numpy()

	if init is None:
		n_clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(y)
	else:
		init = init.cpu().numpy()
		n_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=1, init=init).fit(y)
	res = device(n_clustering.labels_)
	clustering_res = torch.zeros_like(index.float())
	clustering_res[index] = res
	clustering_res = clustering_res.long()
	kmeans_res = torch.zeros_like(group_res) + group_res
	for i in range(c):
		kmeans_res[kmeans_res == i + 1] = -clustering_res[i] - 1
	kmeans_res = - kmeans_res
	center = n_clustering.cluster_centers_
	center = torch.stack([device(np.array(ct)) for ct in center], 0)
	print("\t2. kmeans done")
	return kmeans_res, center

def manul_correct(group_res, center_target, ratio=0.5):
	
	all_n = len(center_target)
	for i, center in enumerate(center_target):
		t = int(group_res.shape[-1] * ratio)
		center = center[:, :t]
		data = group_res[:, :t]
		data = group_res[:, :t][center == 1].unique()
		for c in data:
			if c <= all_n:
				continue
			group_res[group_res == c] = i + 1

	data = group_res[:, :t]
	data = data[data > all_n].unique()
	for c in data:
		group_res[group_res == c] = 0
	return group_res

def clustering(nnet, abt_rep, note_target, group_res, c, n_clusters):
	print("clustering...")
	
	N, T = note_target.shape
	#n_clusters = len(center_target)
	abt_rep = abt_rep.transpose(-1, -2)

	group_res = group_res.long().flatten(0, 2)
	index = torch.ones(c).to(note_target.device) == 1
	clustering_embedding, group_res, index = compute_center(abt_rep, note_target, group_res, index, n_clusters)
	kmeans_res, center = kmeans(group_res, clustering_embedding, index, n_clusters)

	x_ind = (torch.arange(N)[:, None].to(note_target.device) + torch.zeros_like(note_target)).flatten()
	y_ind = (torch.arange(T)[None, :].to(note_target.device) + torch.zeros_like(note_target)).flatten()
	ind = kmeans_res > 0
	x = x_ind[ind]
	y = y_ind[ind]
	c = kmeans_res[ind]
	emb = clustering_embedding[group_res[ind] - 1]

	exported_data = torch.cat([x[:, None], y[:, None], c[:, None], emb], 1)	
	print(exported_data.shape)

	#dc = nnet(center, "inference_dc")
	classify_res = torch.zeros([n_clusters, group_res.shape[0]]).to(center.device)
	for i in range(n_clusters):
		 classify_res[i, kmeans_res == i + 1] = 1

	#for i in range(c):
	#	ind = group_res == - (i + 1)
	#	if ind.sum() > 0:
	#		r = predict((clustering_embedding[i][None, :] * dc[:, :-1]).sum(-1) + dc[:, -1])
	#		for j in range(n_clusters):
	#			if r[j] == 1:
	#				classify_res[j, ind] = 1


	outputs = []
	for i in range(n_clusters):
		outputs.append(classify_res[i].view(N, T))
	print("clustering done")
	return outputs, center, exported_data
		

def flatten_target(target):
	target = target.transpose(0, 1)
	x_ind = (torch.arange(target.shape[0])[:, None].to(target.device) + torch.zeros_like(target)).flatten()
	y_ind = (torch.arange(target.shape[1])[None, :].to(target.device) + torch.zeros_like(target)).flatten()
	data = target.flatten(0, 1)
	index = data > 0
	return torch.stack([x_ind[index], y_ind[index]], 0).cpu().numpy()


def smooth(x):
	spec_len = x.shape[-1] - SMOOTH_WIN + 1
	res = torch.zeros_like(x[:, :spec_len])
	win = torch.exp(-((torch.arange(SMOOTH_WIN) - SMOOTH_WIN//2) / 2.25)**2)
	win = win / win.sum()
	for i in range(SMOOTH_WIN):
		res[:, :] += win[i] * x[:, i : i + spec_len]
	x[:, SMOOTH_WIN//2 : SMOOTH_WIN//2 + spec_len] = res
	return x

def filter_note_target(note_target, onset_target):
	TOL = 5
	NOL = 63

	onset_target = predict(onset_target, 0.2)
	note_target = predict(note_target, 0.5)
	onset_target = note_target * onset_target

	
	print("onset", onset_target.sum())
	
	res = torch.zeros_like(onset_target)			
	for i in range(note_target.shape[0]):
		in_seg = False
		out_seg = 0
		note_seg = 0
		for j in range(note_target.shape[-1]):
			if onset_target[i, j] == 1:
				in_seg = True
				out_seg = TOL

			if in_seg:
				note_seg = 0
				if note_target[i, j] == 1:
					res[i, j - TOL + out_seg: j + 1] = 1
					out_seg = TOL
				else:
					out_seg -= 1
					if out_seg == 0:
						in_seg = False
			elif note_target[i, j] == 1:
				note_seg += 1
			else:
				note_seg = 0

			if note_seg > NOL:
				res[i, j + 1 - note_seg: j + 1] = 1

	note_target = res
	res = torch.zeros_like(onset_target)
	for i in range(note_target.shape[0]):
		seg_len = 0
		for j in range(note_target.shape[-1]):
			if onset_target[i, j] == 1:
				seg_len += 1
			else:
				if seg_len > 0:
					res[i, j - seg_len] = 1
				seg_len = 0
	onset_target = res
	return note_target, onset_target


def predict_note(note_prob, onset_prob):
	note_target, onset_target = filter_note_target(note_prob, onset_prob)
	return note_target, onset_target


def inference_rep(nnet, mix_spec, spec_len):
	seg_frame = int(SEG_SEC * FRAMES_PER_SEC)
	segs_num = mix_spec.shape[-1] // seg_frame
	outputs = []
	for i in range(segs_num):
		st = i * seg_frame
		ed = (i + 1) * seg_frame
		spec_in = mix_spec[None, :, st:ed]
		output = nnet(spec_in, "inference_rep")
		outputs.append(output)
	params = [torch.cat([o[i].squeeze(0) for o in outputs], -1) for i in range(4)]
	for i in range(2, 4):
		params[i][:, spec_len:] = 0
	rep, abt_rep, note_prob, onset_prob = params
	note_target,	onset_target = predict_note(note_prob, onset_prob)
	params = (rep, abt_rep, note_target, onset_target)
	return params


def scale(mean, std, c=1):
	def scale_fn(x):
		_mean = x.mean()
		_std = x.std()
		x = (x - _mean)/ _std * std + mean
		return x / c, _mean, _std
	return scale_fn


def merge(x, y, norm=True):
	res = []
	for i, v in enumerate(x):
		if i == 0 or not y[i] == y[i - 1]:
			res.append(v)
		else:
			v += res[-1]
			res[-1] = v
	
	if norm:
		for i, v in enumerate(res):
			res[i][v > 1] = 1	
	print(f"Pre: merge tracks {len(x)} -> {len(res)}")
	return res


def parse_interval(x):
	intervals = []
	pitches = []
	for i in range(x.shape[0]):
		st = -1
		for j in range(x.shape[1] + 1):
			if j == x.shape[1] or x[i, j] == 0:
				if st > -1:
					intervals.append([st * 1. / FRAMES_PER_SEC, j * 1. / FRAMES_PER_SEC])
					pitches.append(i)
				st = -1
			elif st == -1:
				st = j
	res = np.concatenate([np.array(intervals), np.array(pitches)[:, None]], -1)
	return res

def instrument_aware_ref(data):
	res = []
	for i, d in enumerate(data):
		d[:, 2] += i * 1000
		res.append(d)
	res = np.concatenate(res, 0)
	return res[:, :2], res[:, 2]


def preprocess_data(mix, scale_fn, sources_num):
	_, mean, std = scale_fn(mix)
	wav_len = mix.shape[-1]
	seg_frame = int(SEG_SEC * FRAMES_PER_SEC)
	mixture_spec, cos, sin = wav_2_spec(mix[None, :])
	spec_len = mixture_spec.shape[-1]
	segs_num = (spec_len + seg_frame - 1) // seg_frame
	pad_len = segs_num * seg_frame - spec_len
	mix_spec, cos, sin = wav_2_spec(mix[None, :])
	sin = sin.squeeze(0)
	cos = cos.squeeze(0)
	mix_spec = F.pad(mix_spec.squeeze(0), (0, pad_len), "constant", 0)
	mix_scale_fn = scale(mean, std, sources_num)
	return mix_spec, cos, sin, mix_scale_fn, spec_len, wav_len, pad_len, segs_num, seg_frame

def transcribe_and_separate(nnet, mix, sources_num, is_clustering):
	mix_spec, cos, sin, mix_scale_fn, spec_len, wav_len, pad_len, segs_num, seg_frame = preprocess_data(mix, scale(0., 1), sources_num)
	rep, abt_rep, note_target, onset_target = inference_rep(nnet, mix_spec, spec_len)
	print(abt_rep.shape, note_target.shape)
	if is_clustering:
		group_res, c = group(note_target, onset_target, abt_rep, sources_num)

		N, T = group_res.shape
		targets_cluster, center, exported_data = clustering(nnet, abt_rep, note_target, group_res[:, None], c, sources_num)

	else:
		targets_cluster = [note_target] * sources_num

	targets_pred = []
	est_wavs = []
	for i, target in enumerate(targets_cluster):
		outputs = []
		for j in range(segs_num):
			st = j * seg_frame
			ed = (j + 1) * seg_frame
			rec_target = targets_cluster[i][None, :, st : ed]
			sep, est_note = nnet((rep[None, :, st:ed], rec_target), "inference_spec")
			outputs.append([sep, est_note])

		cat_outputs = []
		for j in range(2):
			cat_outputs.append(torch.cat([out[j] for out in outputs], -1).transpose(0, -1)[:spec_len].transpose(0, -1))
		sep_wav = [spec_2_wav(out.squeeze(0), cos, sin, wav_len, phase=True)[None, :] for out in cat_outputs[:1]][0]
		targets_pred.append([c.squeeze(0) for c in cat_outputs[1:]])
		est_wavs.append(mix_scale_fn(sep_wav)[0])
	note_pred = [pred[0] for pred in targets_pred]

	if is_clustering:
		return est_wavs, note_pred, center.cpu().numpy(), abt_rep.transpose(1, 2).transpose(0, 1).cpu().numpy(), exported_data.cpu().numpy()
	return est_wavs, note_pred,

def permute(est_wavs, est_notes, notes):
	res = []
	for i in range(len(est_notes)):
		max_ratio = -1
		ind = notes[i] == 1
		for j in range(notes.shape[0]):
			ratio = est_notes[j][ind].sum() / est_notes[j].sum()
			if ratio > max_ratio:
				max_ratio = ratio
				max_c = j
		res.append(max_c)
	res_wavs = []
	res_notes = []
	for c in res:
		res_wavs.append(est_wavs[c])
		res_notes.append(est_notes[c])
	return res_wavs, torch.stack(res_notes, 0)

def align(x, y):
	frame_len = y[-1].shape[-1]
	if frame_len < x.shape[-1]:
		return x[:, :, :frame_len]
	return F.pad(x, (0, frame_len - x.shape[-1]), "constant", 0)

def fit_pca(x):
	pca = PCA(n_components=2)
	y = pca.fit_transform(x)
	return y



