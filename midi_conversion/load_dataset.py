#%%

import pandas as pd
from mido import MidiFile
import muspy
import torch
import numpy as np

path = 'D:/PSIML/datasets/maestro-v3.0.0/'
dataset_path = path + 'maestro-v3.0.0.csv'
dataframe = pd.read_csv(dataset_path)
filenames = dataframe['midi_filename'].tolist()
print(len(filenames))

dataset = []

i = 0
for fname in filenames:
    music = muspy.read_midi(path + fname, backend='pretty_midi')
    music_pitch = muspy.to_pitch_representation(music, use_hold_state=True, dtype=int)
    dataset.append(music_pitch.squeeze())
    print(music_pitch.shape)
    #np.concatenate(dataset, music_pitch.squeeze())
    #dataset.append
    
    if i >= 10: break
    i+=1

#%%
#dataset_np = np.array(dataset, dtype=int)
#new_dataset = np.array(dataset, dtype=np.ndarray).flatten()
#print(new_dataset)
# %%
new_dataset = np.concatenate(dataset, axis=0)
print(new_dataset.shape)
# %%
