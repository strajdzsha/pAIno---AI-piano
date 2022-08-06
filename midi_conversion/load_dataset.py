import pandas as pd
import music21
from mido import MidiFile
import muspy
import numpy as np
import os

END_TOKEN = 420

# major conversions
majors = dict([("G#", 4), ("A-", 4), ("A", 3), ("A#", 2), ("B-", 2), ("B", 1), ("B#", 0), ("C", 0),
("C#", -1), ("D-", -1), ("D", -2), ("D#", -3), ("E-", -3), ("E", -4), ("E#", -5), ("F", -5), ("F#", 6), ("G-", 6), ("G", 5)])

minors = dict([("G#", 1), ("A-", 1), ("A", 0), ("A#", -1), ("B-", -1), ("B", -2), ("B#", -3), ("C", -3), 
("C#", -4), ("D-", -4), ("D", -5), ("D#", 6), ("E-", 6), ("E", 5), ("E#", 4), ("F", 4), ("F#", 3), ("G-", 3), ("G", 2)])

path = 'C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\midi_conversion\\dataset\\'
#dataset_path = path + 'maestro-v3.0.0.csv'
#dataframe = pd.read_csv(dataset_path)
#filenames = dataframe['midi_filename'].tolist()
#print(len(filenames))

dataset = []
filenames = os.listdir("midi_conversion/dataset")

for fname in filenames:
    music = muspy.read_midi(path + fname, backend='pretty_midi')
    music.adjust_resolution(4)
    
    score = music21.converter.parse(path + fname)
    key = score.analyze('key')

    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
        
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    try:
        music_pitch = muspy.to_pitch_representation(music.transpose(halfSteps), use_hold_state=True, dtype=int)
    except:
        print("Bad file.")
    dataset.append(music_pitch.squeeze())
    dataset.append([END_TOKEN])

new_dataset = np.concatenate(dataset, axis=0)
#np.save("Dataset_mini_with_end_tokens_lower_res.npy", new_dataset)