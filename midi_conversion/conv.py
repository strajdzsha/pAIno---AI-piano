import pandas as pd
from mido import MidiFile
import muspy
import numpy as np
import os


def convertToPitch(filepath):
    music = muspy.read_midi(filepath, backend='pretty_midi')
    music_pitch = muspy.to_pitch_representation(music, use_hold_state=True, dtype=int)
    return music_pitch.squeeze()

def writeMidi(music_pitch, filepath):
    new_midi = muspy.from_pitch_representation(music_pitch, resolution=24, program=0, is_drum=False, use_hold_state=True, default_velocity=100)
    new_midi_file = muspy.write_midi(filepath, new_midi, backend='pretty_midi')

idx = 0
for file in os.listdir("out/model outputs/"):
    music_pitch = np.load("out/model outputs/output_mini_" + str(idx) + ".npy")
    writeMidi(music_pitch, "out/music/delete_this_" + str(idx) + ".midi")
    idx += 1
    if idx > 7:
        break
