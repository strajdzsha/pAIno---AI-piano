import pandas as pd
from mido import MidiFile
import muspy
import numpy as np
import os

END_TOKEN = 420

path = 'C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\midi_conversion\\dataset\\'
#dataset_path = path + 'maestro-v3.0.0.csv'
#dataframe = pd.read_csv(dataset_path)
#filenames = dataframe['midi_filename'].tolist()
#print(len(filenames))

dataset = []
filenames = os.listdir("midi_conversion/dataset")

for fname in filenames:
    music = muspy.read_midi(path + fname, backend='pretty_midi')
    try:
        music_pitch = muspy.to_pitch_representation(music, use_hold_state=True, dtype=int)
    except:
        print("Bad file.")
    dataset.append(music_pitch.squeeze())
    dataset.append([END_TOKEN])

new_dataset = np.concatenate(dataset, axis=0)
np.save("Dataset_mini_with_end_tokens.npy", new_dataset)