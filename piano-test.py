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

from minGPT.mingpt.model import GPT
from minGPT.mingpt.trainer import Trainer
from minGPT.mingpt.utils import set_seed, setup_logging, CfgNode as CN

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

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
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
    C.model.model_type = 'gpt-mini'

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

    full_path_to_training_text_file = "C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\Dataset.txt" 
    dataset_arr = np.loadtxt(full_path_to_training_text_file)
    #text = open(full_path_to_training_text_file, 'r').read() 
    train_dataset = CharDataset(config.data, dataset_arr) 


    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    trainer = Trainer(config.trainer, model, train_dataset)

    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        
        if trainer.iter_num % 5000 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                rand_int = np.random.randint(0, 10)
                contexts = [[20, 89, 89, 89], [30, 20, 40, 50], [20], [70, 89], [30, 89, 89, 89, 89, 89, 89, 89, 89, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 42], [54, 89, 89, 89, 89, 89, 60, 89, 89, 89, 89], [40], [50], [10], [88], [13, 13, 13, 13]]
                context = contexts[rand_int]
                x = torch.tensor(context, dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                print(y)
                with open("output_" + str(trainer.iter_num //5000) + ".npy", "wb") as f:
                    np.save(f, y.to("cpu").numpy())
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model" + str(trainer.iter_num // 5000) + ".pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()