import pandas as pd
import music21
from mido import MidiFile
import muspy
import numpy as np
import os
from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile
import glob

TICKS_PER_BIT = 480

pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)

root_dataset = "D:\\PSIML\\datasets\\"
midi = MidiFile(root_dataset + "maestro-v3.0.0\\2004\MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_05_Track05_wav.midi")

root = "D:\\PSIML\\code\\pAIno---AI-piano\\Evaluation\\generated_samples\\"
for filename in os.listdir(root):

    piano_tokens = np.load(root + filename).tolist()

    converted_back_midi = tokenizer.tokens_to_midi([piano_tokens], get_midi_programs(midi))
    converted_back_midi.dump(root + filename + '.midi')