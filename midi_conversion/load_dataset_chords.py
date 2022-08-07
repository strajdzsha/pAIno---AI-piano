import pandas as pd
import music21
from mido import MidiFile
import muspy
import numpy as np
import os
from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile
import glob

END_TOKEN = 420

# major conversions
majors = dict([("G#", 4), ("A-", 4), ("A", 3), ("A#", 2), ("B-", 2), ("B", 1), ("B#", 0), ("C", 0),
("C#", -1), ("D-", -1), ("D", -2), ("D#", -3), ("E-", -3), ("E", -4), ("E#", -5), ("F", -5), ("F#", 6), ("G-", 6), ("G", 5)])

minors = dict([("G#", 1), ("A-", 1), ("A", 0), ("A#", -1), ("B-", -1), ("B", -2), ("B#", -3), ("C", -3), 
("C#", -4), ("D-", -4), ("D", -5), ("D#", 6), ("E-", 6), ("E", 5), ("E#", 4), ("F", 4), ("F#", 3), ("G-", 3), ("G", 2)])

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

path = 'C:\\Users\\psiml8\\VS projects\\pAIno---AI-piano\\midi_conversion\\dataset\\'

dataset = []
filenames = os.listdir("midi_conversion/dataset/")
folder = 'C:/Users/psiml8/Downloads/maestro-v3.0.0-midi/maestro-v3.0.0/**/*'
list_all_midi = glob.glob(folder)


for i, fname in enumerate(list_all_midi):
    #music_pitch = muspy.to_pitch_representation(music.transpose(halfSteps), use_hold_state=True, dtype=int)
    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)
    midi = MidiFile(fname)
    # Converts MIDI to tokens, and back to a MIDI
    tokens = tokenizer.midi_to_tokens(midi)
    # Converts just a selected track
    tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
    piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])
    dataset.append(piano_tokens)
    dataset.append([END_TOKEN])

    if i==1000: break
    


new_dataset = np.concatenate(dataset, axis=0)
print(len(new_dataset))
np.save("Dataset_maestro_with_chords.npy", new_dataset)