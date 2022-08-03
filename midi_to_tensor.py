#from mido import MidiFile
import muspy
import torch

midi_path = 'bwv40.3.mxl.mid'
music = muspy.read(midi_path)
music_pitch = muspy.to_pitch_representation(music, use_hold_state=False, dtype=int)

print(music_pitch)
print(music_pitch[0])
print(len(music_pitch))

music_tensor = torch.tensor(music_pitch)
print(music_tensor)