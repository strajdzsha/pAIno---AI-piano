import muspy
from mido import MidiFile
from mido import MidiTrack, Message

data = muspy.datasets.Music21Dataset(composer='bach')


root = "D:/PSIML/code/testing/dataset/bach"

n = 1
mid = MidiFile(root + str(n) + ".midi")

print("Type: ", mid.type)
print("Tracks: ", len(mid.tracks))

for msg in mid.tracks[0]:
    print(msg)

print("Midi file length: ")
print(len(mid.tracks[1]))
#print(mid.tracks[4])

track_1_midi = MidiFile()
track_1_midi.tracks.append(mid.tracks[0])
track_1_midi.tracks.append(mid.tracks[1])
print(track_1_midi)

track_1_midi.save("track_1_midi.mid")


mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=64, velocity=64, time=32))
track.append(Message('note_off', note=64, velocity=127, time=32))

mid.save('D:/PSIML/code/testing/dataset/new_song.mid')
