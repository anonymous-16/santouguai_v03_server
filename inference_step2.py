import numpy as np
import torchaudio
import json
import torch
import os

import src
from utils.utilities import mkdir, numpy_2_midi
from models.models import DisentanglementModel
from conf.feature import SAMPLE_RATE
from evaluation.inference_utils import transcribe_and_separate, align, fit_pca

def numpy_2_json(center, emb, output_path, spec_len):
    center = center.tolist()
    dt = np.dtype([('name', 'S10'), ('age', int)])
    a = np.array([("raju", 21), ("anil", 25), ("ravi", 17), ("amar", 27)], dtype=dt)
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
    data = emb[:, :3].tolist()
    json_data = {"sourcesNum": len(center), "center": center, "color": color,
                 "pianoRoll": {"data": data, "border": [0, min_p, max_sec, max_p]},
                "timbre": {"data": timbre, "border": border_t}}

    with open(output_path, 'w') as f:
        json.dump(json_data, f)

def inference(model_path, audioData, sources_num, file_type):
    nnet = DisentanglementModel().cuda()
    nnet.load_state_dict(torch.load(model_path))
    audioData = audioData.cuda().squeeze(0)
    output_folder = "output"
    mkdir(output_folder)
    with torch.no_grad():
        est_wavs, note_preds, center, rep, exported_data = transcribe_and_separate(nnet, audioData, sources_num,
                                                                              is_clustering=True)
        emb = fit_pca(np.concatenate([center, exported_data[:, 3:]], 0))
        emb = emb * 1000
        center = emb[:center.shape[0]]
        emb = np.concatenate([exported_data[:, :3], emb[center.shape[0]:]], -1)
        spec_len = note_preds[0].shape[-1]
        numpy_2_json(center, emb, os.path.join(output_folder, "data.json"), spec_len)
        np.save(os.path.join(output_folder, "rep.npy"), rep)
        numpy_2_midi([n.cpu().numpy() for n in note_preds], os.path.join(output_folder, "pred.mid"))
        for i, wav in enumerate(est_wavs):
            torchaudio.save(os.path.join(output_folder, f"{i}.{file_type}"), wav.float().cpu(), SAMPLE_RATE)
    return output_folder

if __name__ == "__main__":
    data_path = "input_data/metadata"
    with open(data_path, "r") as f:
        metadata = json.load(f)
    file_type = metadata["audioType"]
    sources_num = metadata["sourcesNum"]
    audio_path = "input_data/mix.wav" if str.endswith(file_type, "wav") else "input_data/mix.mp3"
    model_path = "model_weights"
    audio_data, src = torchaudio.load(audio_path)
    file_type = "wav" if str.endswith(file_type, "wav") else "mp3"
    inference(model_path, audio_data, sources_num, file_type)

