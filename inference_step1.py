import numpy as np
import torchaudio
import json
import torch
import os

import src

from utils.utilities import mkdir, numpy_2_midi
from models.models import DisentanglementModel
from conf.feature import SAMPLE_RATE
from evaluation.inference_utils import transcribe_and_separate, generate_output


def inference(model_path, audioData, sources_num, file_type):
    nnet = DisentanglementModel().cuda()
    nnet.load_state_dict(torch.load(model_path))
    audioData = audioData.cuda().squeeze(0)
    output_folder = "output_step1"
    mkdir(output_folder)
    est_wavs, note_preds, center, rep, exported_data = transcribe_and_separate(nnet, audioData, sources_num)
    generate_output(output_folder, center, exported_data, note_preds, est_wavs)

if __name__ == "__main__":
    data_path = "input_data/metadata"
    with open(data_path, "r") as f:
        metadata = json.load(f)
    file_type = metadata["audioType"]
    sources_num = metadata["sourcesNum"]
    audio_path = "input_data/mix.wav" if str.endswith(file_type, "wav") else "input_data/mix.mp3"
    model_path = "model_weights"
    audio_data, sr = torchaudio.load(audio_path)
    if not sr == SAMPLE_RATE:
        audio_data = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio_data)
    file_type = "wav" if str.endswith(file_type, "wav") else "mp3"
    inference(model_path, audio_data, sources_num, file_type)

