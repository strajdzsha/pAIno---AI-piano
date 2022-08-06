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
from midi_conversion.load_dataset import END_TOKEN

from minGPT.mingpt.model import GPT
from minGPT.mingpt.trainer import Trainer
from minGPT.mingpt.utils import set_seed, setup_logging, CfgNode as CN


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
    C.model.model_type = 'gpt2'

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
        C.block_size = 96
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
            chunk[i:] = 129 # 129 is hold note
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


def writeMidi(music_pitch, filepath):
    new_midi = muspy.from_pitch_representation(music_pitch, resolution=24, program=0, is_drum=False, use_hold_state=True, default_velocity=100)
    new_midi_file = muspy.write_midi(filepath, new_midi, backend='pretty_midi')


def loadRandomSong():
    path = "midi_conversion\\dataset\\"
    filenames = os.listdir(path)
    randint = np.random.randint(0, len(filenames))

    fname = filenames[randint]
    print(fname)
    music = muspy.read_midi(path + fname, backend='pretty_midi')
    music_pitch = muspy.to_pitch_representation(music, use_hold_state=True, dtype=int)

    return music_pitch[:300]



config = get_config()

full_path_to_training_text_file = "C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\Dataset_mini_with_end_tokens.npy" 
dataset_arr = np.load(full_path_to_training_text_file)
train_dataset = CharDataset(config.data, dataset_arr) 


config.model.vocab_size = train_dataset.get_vocab_size()
config.model.block_size = train_dataset.get_block_size()
model = GPT(config.model)

model.load_state_dict(torch.load("out\\models\\model24.pt"))

trainer = Trainer(config.trainer, model, train_dataset)

model.eval()
n = 0

with torch.no_grad():
    context = loadRandomSong().squeeze().tolist() # pitch representation of prompt
    context = [train_dataset.stoi[x] for x in context] # going form pitch representation to something model would train on
    x = torch.tensor(context, dtype=torch.long)[None,...].to(trainer.device)
    y = model.generate(x, 1000, temperature=1.0, do_sample=True, top_k=10)[0]
    converted_y = np.array([train_dataset.itos[c.item()] for c in y]) # reverting back to pitch representation
    print(converted_y)
    with open("out/model outputs/generated_" + str(n) + ".npy", "wb") as f:
        np.save(f, converted_y)

music_pitch = np.load("out/model outputs/generated_" + str(n) + ".npy")
writeMidi(music_pitch, "out/music/generated_" + str(n) + ".midi")