import pandas as pd
from mido import MidiFile
import muspy
import numpy as np
import os

'''
music_pitch = np.array([20, 89, 30, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 47, 42, 39, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 51, 89, 89, 49, 89, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 49, 46, 47, 89, 89, 47, 37, 89, 89, 51, 89, 89, 89, 51,
        89, 89, 49, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 88, 88, 88, 56, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 88, 88, 88, 88, 88, 53, 89, 42, 89, 89, 89, 89, 89,
        89, 89, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 49, 89, 89, 89, 89, 89, 53, 89,
        89, 89, 89, 89, 89, 89, 47, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
        89, 53, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 51, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 49, 89, 89, 89, 89, 89,
        89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
        88, 88, 88, 88, 88, 88, 47, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89])
        '''


def convertoToPitch(filepath):
    music = muspy.read_midi(filepath, backend='pretty_midi')
    music_pitch = muspy.to_pitch_representation(music, use_hold_state=True, dtype=int)
    return music_pitch.squeeze()

def writeMidi(music_pitch, filepath):
    new_midi = muspy.from_pitch_representation(music_pitch, resolution=24, program=0, is_drum=False, use_hold_state=True, default_velocity=100)
    new_midi_file = muspy.write_midi(filepath, new_midi, backend='pretty_midi')

idx = 0
for file in os.listdir("out/model outputs/"):
    music_pitch = np.load("out/model outputs/output_" + str(idx) + ".npy")
    music_pitch += 40
    writeMidi(music_pitch, "out/music/output_" + str(idx) + ".midi")
    idx += 10
    if idx > 190:
        break