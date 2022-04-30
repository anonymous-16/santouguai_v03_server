import numpy as np
import torchaudio
import json
import torch
import os

import src
from utils.torch_utilities import wav_2_spec
from utils.utilities import mkdir, numpy_2_midi, read_lst, freq_2_note
from models.models import DisentanglementModel
from conf.feature import SAMPLE_RATE, NOTES_NUM, FRAMES_PER_SEC
from evaluation.inference_utils import extract_rep, torch_device, extract_center


def inference(model_path, note_data, audio_data):
    nnet = DisentanglementModel().cuda()
    nnet.load_state_dict(torch.load(model_path))

    mix = audio_data.squeeze(0)
    center = extract_center(nnet, mix, note_data, len(note_data))
    print(center.shape)
    return center

def sec_2_frame(onset, offset=0):
    return round((float(onset) + float(offset)) * FRAMES_PER_SEC)

def load_note_data(note_path, spec_len):
    targets = []
    for source_path in note_path:
        target = torch.zeros([NOTES_NUM - 1, spec_len])
        for path in source_path:
            notes = read_lst(path, "\t\t")
            for note in notes:
                target[freq_2_note(note[1]) - 1, sec_2_frame(note[0]) : sec_2_frame(note[0], note[2])] = 1
        targets.append(target)
    mask = sum(targets)
    mask[mask > 1] = 0
    targets = [target * mask for target in targets]
    return targets

def scan_dataset(folder):
    songs = os.listdir(folder)
    for song in songs:
        song_folder = os.path.join(folder, song)
        if os.path.isfile(song_folder):
            continue
        print(f"Processing {song}...")
        mix_path = os.path.join(song_folder, f"AuMix_{song}.wav")
        notes = {}
        for path in os.listdir(song_folder):
            if str.startswith(path, "Notes"):
                instr = path.split("_")[2]
                if instr not in notes:
                    notes[instr] = []
                notes[instr].append(os.path.join(song_folder, path))
        process(song, mix_path, notes)
        print(f"Processing done.")

def process(song_folder, mix_path, notes):
    output_folder = "timbre_db"
    mkdir(output_folder)
    audio_data, sr = torchaudio.load(mix_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[0]
    spec_len = int((audio_data.shape[-1] / sr + 1) * FRAMES_PER_SEC)
    tags = [tag for tag in notes]
    notes = [notes[tag] for tag in tags]
    note_data = load_note_data(notes, spec_len)
    if not sr == SAMPLE_RATE:
        audio_data = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio_data)
    center = inference(model_path, note_data, audio_data)
    res = {}

    for i, tag in enumerate(tags):
        res[tag] = center[i].cpu().numpy().tolist()
    with open(os.path.join(output_folder, f"{song_folder}.json"), "w") as f:
        json.dump(res, f)

if __name__ == "__main__":
    #model_path = "model_weights"
    #dataset_folder = "D:\\linux\\wei_workspace\\dataset\\URMP\Dataset"
    #scan_dataset(dataset_folder)

    folder = "timbre_db"
    files = os.listdir(folder)
    res = []
    for path in files:
        path = os.path.join(folder, path)
        with open(path, "r") as f:
            data = json.load(f)
        for tag in data:
            res.append({"tag": tag, "embedding": data[tag]})
    with open("timbre_db.json", "w") as f:
        json.dump(res, f)

