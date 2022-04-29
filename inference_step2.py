import numpy as np
import torchaudio
import json
import torch
import os

import src
from utils.torch_utilities import wav_2_spec
from utils.utilities import mkdir, numpy_2_midi
from models.models import DisentanglementModel
from conf.feature import SAMPLE_RATE, NOTES_NUM
from evaluation.inference_utils import extract_rep, refine, generate_output, torch_device, scale

def json_2_numpy(x, sources_num, spec_len):
    pos = torch_device(np.array(x["data"]))
    c = torch_device(np.array(x["color"])) - 1
    cluster_targets = []
    for i in range(sources_num):
        zero_target = torch.zeros([spec_len, NOTES_NUM - 1]).to(pos.device).flatten()
        ind = c == i + 1
        ind = pos[ind]
        nind = ind[:, 0] * (NOTES_NUM - 1) + (NOTES_NUM - 1 - ind[:, 1])
        zero_target[nind.long()] = 1
        cluster_targets.append(zero_target.view(spec_len, NOTES_NUM - 1))
    return cluster_targets

def inference(model_path, note_data, audio_data, sources_num, file_type="wav"):
    nnet = DisentanglementModel().cuda()
    nnet.load_state_dict(torch.load(model_path))

    output_folder = "output_step2"
    mkdir(output_folder)

    mix = audio_data.squeeze(0)
    rep, abt_rep, spec_len, wav_len, cos, sin = extract_rep(nnet, mix, sources_num)
    target = torch.zeros([NOTES_NUM - 1, rep.shape[-1]]).to(rep.device)
    _, mean, std = scale(0., 1)(mix)
    mix_scale_fn = scale(mean, std, sources_num)
    est_wavs, note_preds, center, exported_data = \
        refine(nnet, note_data, rep, abt_rep, cos, sin, wav_len, spec_len, mix_scale_fn, sources_num, target)
    generate_output(output_folder, center, exported_data, note_preds, est_wavs)
    return output_folder

if __name__ == "__main__":
    data_path = "revised_data/"
    json_path = os.path.join(data_path, "data.json")
    audio_path = os.path.join(data_path, "mix.wav")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    note_data = json_data["data"]
    sources_num = json_data["sourcesNum"]
    model_path = "model_weights"
    audio_data, sr = torchaudio.load(audio_path)
    if not sr == SAMPLE_RATE:
        audio_data = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio_data)
    inference(model_path,  note_data, audio_data, sources_num)

