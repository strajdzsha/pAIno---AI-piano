import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import time
import math
import os

import muspy

import tqdm.auto
import matplotlib
import matplotlib.pyplot as plt

from minGPT.mingpt.model import GPT
from minGPT.mingpt.trainer import Trainer
from minGPT.mingpt.utils import set_seed, setup_logging, CfgNode as CN

from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile

END_TOKEN = 420

pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
root_dataset = "D:\\PSIML\\datasets\\"
midi = MidiFile(root_dataset + "maestro-v3.0.0\\2004\MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_05_Track05_wav.midi")


def get_config():

    C = CN()

    # system
    C.system = CN()
    #C.system.seed = 3407
    C.system.work_dir = './out/models'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 6e-4 # the model we're using is so small that we can go a bit faster
    C.trainer

    return C


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 512
        return C

    def __init__(self, config, data):
        self.config = config

        self.END_TOKEN = 420 #token that tells we encoutered end of the song

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        # print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size - 1 # -1 because of end token
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        if self.END_TOKEN in chunk:
            i = np.where(chunk == END_TOKEN)
            i = i[0].squeeze()
            chunk[i:] = chunk[i-2] # 129 is hold note
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


def writeMidi(piano_tokens, filepath):
    converted_back_midi = tokenizer.tokens_to_midi([piano_tokens], get_midi_programs(midi))
    converted_back_midi.dump(filepath + '.midi')


def loadRandomSong():
    path = "D:\\PSIML\\datasets\\maestro-v3.0.0\\2004\\"
    filenames = os.listdir(path)
    randint = np.random.randint(0, len(filenames))

    fname = filenames[randint]
    print(fname)
    midi = MidiFile(path + fname)

    tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
    piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])

    return piano_tokens[:150]



config = get_config()
config.data.block_size = 512
config.trainer.learning_rate = 0.001

full_path_to_training_text_file = "C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\Dataset_maestro_with_chords.npy" 
dataset_arr = np.load(full_path_to_training_text_file)
train_dataset = CharDataset(config.data, dataset_arr) 


config.model.vocab_size = train_dataset.get_vocab_size()
config.model.block_size = train_dataset.get_block_size()
model = GPT(config.model)

model.load_state_dict(torch.load("pAIno---AI-piano\model_0.001_512_best.pt"))

trainer = Trainer(config.trainer, model, train_dataset)

model.eval()
n = 0

with torch.no_grad():
    context = loadRandomSong().squeeze().tolist() # pitch representation of prompt
    context = [train_dataset.stoi[x] for x in context] # going form pitch representation to something model would train on
    x = torch.tensor(context, dtype=torch.long)[None,...].to(trainer.device)
    y = model.generate(x, 1000, temperature=1.0, do_sample=True, top_k=10)[0]
    converted_y = np.array([train_dataset.itos[c.item()] for c in y]) # reverting back to pitch representation


# music_pitch = np.load("out/model outputs/generated_" + str(n) + ".npy")
writeMidi(converted_y, "generated.midi")