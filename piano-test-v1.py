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


import tqdm.auto
import matplotlib
import matplotlib.pyplot as plt
# from midi_conversion.load_dataset import END_TOKEN
END_TOKEN = 420

from minGPT.mingpt.model import GPT
from minGPT.mingpt.trainer import Trainer
from minGPT.mingpt.utils import set_seed, setup_logging, CfgNode as CN
import wandb


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        self.END_TOKEN = 420 #token that tells we encoutered end of the song

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

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
            chunk[i:] = chunk[i-2] # CHANGE THIS PLS !!!!!!!!!!!!!!!!!!!
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


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


if __name__ == '__main__':
    

    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #block_size = 128
    config = get_config()

    full_path_to_training_text_file = "C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\Dataset_maestro_with_chords.npy" 
    dataset_arr = np.load(full_path_to_training_text_file)
    train_dataset = CharDataset(config.data, dataset_arr) 


    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    trainer = Trainer(config.trainer, model, train_dataset)
    
    def batch_end_callback(trainer):

        min_loss = 5

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 100 == 0:
            wandb.log({
                "Loss":trainer.loss.item()
            })
        
        n_train_eval = 2000
        if trainer.iter_num % n_train_eval == 0 :
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...

                context = [1, 204, 53, 105, 124, 207, 37, 101, 126, 208, 53, 104, 130, 210, 41, 104, 125, 213, 44] # pitch representation of prompt
                context = [train_dataset.stoi[x] for x in context] # going form pitch representation to something model would train on
                # print(context)
                x = torch.tensor(context, dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 1000, temperature=1.0, do_sample=True, top_k=10)[0]
                converted_y = np.array([train_dataset.itos[c.item()] for c in y]) # reverting back to pitch representation
                # print(converted_y)
                with open("out/model outputs/output_mini_" + str(trainer.iter_num //n_train_eval) + ".npy", "wb") as f:
                    np.save(f, converted_y)

            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model" + str(trainer.iter_num // n_train_eval) + ".pt")
            torch.save(model.state_dict(), ckpt_path)

            if trainer.loss.item() < min_loss:
                ckpt_path = os.path.join(config.system.work_dir, "model_best.pt")
                torch.save(model.state_dict(), ckpt_path)

                min_loss = trainer.loss.item()
        
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)
    wandb.init(entity='strajdzsha', project = 'lgpt3q')
    trainer.run()