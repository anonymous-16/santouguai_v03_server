import os
import torch
import torchaudio
import json
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

from utils.utilities import mkdir, numpy_2_midi
from utils.torch_utilities import (predict, onehot_tensor, spec_2_wav, wav_2_spec)
from .clustering_utils import clustering_from_preds, fit_center, sample_2_frame, get_exported_data, estimate_target
from conf.feature import *
from conf.inference import *

def align(x, y):
	frame_len = y[-1].shape[-1]
	if frame_len < x.shape[-1]:
		return x[:, :, :frame_len]
	return F.pad(x, (0, frame_len - x.shape[-1]), "constant", 0)

def fit_pca(x):
	pca = PCA(n_components=2)
	y = pca.fit_transform(x)
	cps = pca.components_.T
	mean = pca.mean_
	var = pca.explained_variance_
	return y, {"cps" : cps.tolist(), "mean" : mean.tolist(), "var" : var.tolist()}

def numpy_2_json(center, emb, pca_params, output_path, spec_len):
    center = center.tolist()
    dt = np.dtype([('name', 'S10'), ('age', int)])
    ind = np.lexsort((emb[:, 1], -emb[:, 0]))
    emb = emb[ind]
    emb[:, 0] = 128 - emb[:, 0]
    max_sec = spec_len
    max_p = int(np.max(emb[:, 0])) + 4
    min_p = int(np.min(emb[:, 0])) - 4
    border_t = [float(np.min(emb[:, -2])), float(np.min(emb[:, -1])), float(np.max(emb[:, -2])), float(np.max(emb[:, -1]))]
    ind = np.arange(emb.shape[0])[:, None] + 1
    ind[-1, 0] = -1
    color = (emb[:, 2] + 1).tolist()
    emb = np.concatenate([emb[:, 1:2], emb[:, :1], ind, emb[:, 3:]], 1)
    timbre = np.concatenate([emb[:, -2:], ind], 1).tolist()
    data = emb[:, :4].tolist()
    json_data = {"sourcesNum": len(center), "center": center, "color": color, "pca": pca_params,
                 "pianoRoll": {"data": data, "border": [0, min_p, max_sec, max_p]},
                "timbre": {"data": timbre, "border": border_t}}

    with open(output_path, 'w') as f:
        json.dump(json_data, f)

def generate_output(output_folder, center, exported_data, note_preds, est_wavs):
	emb, pca_params = fit_pca(exported_data[:, 4:])
	emb = emb * 1000
	emb = np.concatenate([exported_data[:, :4], emb], -1)
	spec_len = note_preds[0].shape[-1]
	numpy_2_json(center, emb, pca_params, os.path.join(output_folder, "data.json"), spec_len)
	numpy_2_midi([n.cpu().numpy() for n in note_preds], os.path.join(output_folder, "pred.mid"))
	for i, wav in enumerate(est_wavs):
		torchaudio.save(os.path.join(output_folder, f"{i}.wav"), wav.float().cpu(), SAMPLE_RATE)

def gpu_device(x):
	return torch.from_numpy(x).float().cuda()

def cpu_device(x):
	return x.cpu().numpy()

def num(x):
	return int(x.item())

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
	note_target,onset_target = estimate_target(note_prob, onset_prob)
	params = (rep, abt_rep, note_target, onset_target)
	return params

def scale(mean, std, c=1):
	def scale_fn(x):
		_mean = x.mean()
		_std = x.std()
		x = (x - _mean)/ _std * std + mean
		return x / c, _mean, _std
	return scale_fn

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
	return mix_spec, cos, sin, mix_scale_fn, spec_len, wav_len, pad_len

def transcribe_and_separate(nnet, mix, sources_num):
	mix_spec, cos, sin, mix_scale_fn, spec_len, wav_len, pad_len = preprocess_data(mix, scale(0., 1), sources_num)
	nnet.eval()
	with torch.no_grad():
		rep, abt_rep, note_target, onset_target = inference_rep(nnet, mix_spec, spec_len)
	targets_cluster, exported_data, center = clustering_from_preds(note_target, abt_rep, sources_num)
	with torch.no_grad():
		est_wavs, note_pred = inference_wav(nnet, targets_cluster, rep, cos, sin, wav_len, spec_len, mix_scale_fn)
	return est_wavs, note_pred, cpu_device(center), cpu_device(torch.cat([abt_rep.flatten(0, 1), rep], 0)), cpu_device(exported_data)

def extract_rep(nnet, mix, sources_num):
	mix_spec, cos, sin, mix_scale_fn, spec_len, wav_len, pad_len = preprocess_data(mix, scale(0., 1), sources_num)
	nnet.eval()
	with torch.no_grad():
		rep, abt_rep, _, _ = inference_rep(nnet, mix_spec, spec_len)
	return rep, abt_rep, spec_len, wav_len, cos, sin

def inference_wav(nnet, targets_cluster, rep, cos, sin, wav_len, spec_len, mix_scale_fn):
	targets_pred = []
	est_wavs = []
	seg_frame = int(SEG_SEC * FRAMES_PER_SEC)
	segs_num = (spec_len + seg_frame - 1) // seg_frame
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

	return est_wavs, note_pred

def parse_preds(x):
	return x

def refine(nnet, note_data, rep, abt_rep, cos, sin, wav_len, spec_len, mix_scale_fn, sources_num, target):
	index = np.array(note_data["data"])
	cluster = np.array(note_data["color"]) - 1
	sample_index = np.concatenate([NOTES_NUM - 1 - index[:, 1:2], index[:, :1]], 1)
	key_points = index[:, 3]
	ind = cluster > 0
	sample_index = gpu_device(sample_index[ind])
	cluster = gpu_device(cluster[ind]) - 1
	key_points = gpu_device(key_points[ind])
	cluster_targets, exportd_data, center = fit_center(sample_index, abt_rep, cluster, key_points, sources_num, target)

	nnet.eval()
	with torch.no_grad():
		est_wavs, _ = inference_wav(nnet, cluster_targets, rep, cos, sin, wav_len, spec_len, mix_scale_fn)

	return est_wavs, cluster_targets, cpu_device(center), cpu_device(exportd_data)





