#converts all midi files in the current folder

import music21
import pandas as pd
from mido import MidiFile
import muspy
import numpy as np

#converting everything into the key of C major or A minor

# major conversions
majors = dict([("G#", 4), ("A-", 4), ("A", 3), ("A#", 2), ("B-", 2), ("B", 1), ("B#", 0), ("C", 0),
("C#", -1), ("D-", -1), ("D", -2), ("D#", -3), ("E-", -3), ("E", -4), ("E#", -5), ("F", -5), ("F#", 6), ("G-", 6), ("G", 5)])

minors = dict([("G#", 1), ("A-", 1), ("A", 0), ("A#", -1), ("B-", -1), ("B", -2), ("B#", -3), ("C", -3), 
("C#", -4), ("D-", -4), ("D", -5), ("D#", 6), ("E-", 6), ("E", 5), ("E#", 4), ("F", 4), ("F#", 3), ("G-", 3), ("G", 2)])


path = 'D:/PSIML/datasets/emotions/musgenvae-main/dataset/e2/'
#dataset_path = path + 'maestro-v3.0.0.csv'
#dataframe = pd.read_csv(dataset_path)
#filenames = dataframe['midi_filename'].tolist()
#print(len(filenames))

dataset = []
filenames = ["bwv4.8.mxl.mid", "bwv5.7.mxl.mid"]
i = 0
for fname in filenames:

    score = music21.converter.parse(path + fname)
    key = score.analyze('key')

    print(key)
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
        
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    print(halfSteps)
    music = muspy.read_midi(path + fname, backend='pretty_midi')
    #music = music.transpose(2)
    music_pitch = muspy.to_pitch_representation(music.transpose(halfSteps), use_hold_state=True, dtype=int)
    
    back_to_midi = muspy.from_pitch_representation(music_pitch, resolution=24, program=0, is_drum=False, use_hold_state=True, default_velocity=64)
    new_midi_file = muspy.write_midi('new_midi_new' + str(i) + ".midi", back_to_midi, backend='pretty_midi')
    dataset.append(music_pitch.squeeze())

    i+=1

    if (i==5): break

new_dataset = np.concatenate(dataset, axis=0)
print(new_dataset.shape)