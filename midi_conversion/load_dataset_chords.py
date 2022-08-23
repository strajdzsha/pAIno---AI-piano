import pandas as pd
from mido import MidiFile
import numpy as np
import os
from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile
import glob

END_TOKEN = 420

''' Converting midi files to REMI representation that could later be fed into model. We used Maestro dataset
    although you can use any midi dataset you want.'''


# Our parameters; these are really default parameters specified by MIDITOK documentation https://github.com/Natooz/MidiTok
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)


# We used maestro dataset to train our model; it contains ~1k midi files
dataset = []
folder = '/maestro-v3.0.0-midi/maestro-v3.0.0/**/*' 
list_all_midi = glob.glob(folder)


for i, fname in enumerate(list_all_midi):
    # REMI interpretation of midi files, that would be later fed into model
    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True) 
    midi = MidiFile(fname)

    # Converts MIDI to tokens, and back to a MIDI
    tokens = tokenizer.midi_to_tokens(midi)

    # Converts just a selected track
    tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
    piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])
    dataset.append(piano_tokens)

    # Adding END_TOKEN to end of a song; will later be used when creating batches 
    dataset.append([END_TOKEN])

    if i==1000: break
    


new_dataset = np.concatenate(dataset, axis=0)
print(len(new_dataset))
np.save("dataset.npy", new_dataset)