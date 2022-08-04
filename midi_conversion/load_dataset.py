import pandas as pd
from mido import MidiFile
import muspy
import numpy as np

path = 'D:/PSIML/datasets/maestro-v3.0.0/'
dataset_path = path + 'maestro-v3.0.0.csv'
dataframe = pd.read_csv(dataset_path)
filenames = dataframe['midi_filename'].tolist()
print(len(filenames))

dataset = []

for fname in filenames:
    music = muspy.read_midi(path + fname, backend='pretty_midi')
    music_pitch = muspy.to_pitch_representation(music, use_hold_state=True, dtype=int)
    dataset.append(music_pitch.squeeze())
    #print(music_pitch.shape)

new_dataset = np.concatenate(dataset, axis=0)
print(new_dataset.shape)