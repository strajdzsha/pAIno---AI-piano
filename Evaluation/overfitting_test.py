import time
from statistics import mode
import numpy as np
import difflib
import matplotlib.pyplot as plt

dataset = np.load('D:/PSIML/code/pAIno---AI-piano/Dataset_maestro_with_chords.npy')
model_output = np.load('D:/PSIML/code/pAIno---AI-piano/Evaluation/output_mini_40.npy')
song = np.array([1, 204, 53, 105, 124, 207, 37, 101, 126, 208, 53, 104, 130, 210, 41, 104, 125, 213, 44])
#model_output = dataset[len(dataset)//2 : len(dataset)//2 + 1000]
#model_output = song
cor = np.correlate(dataset, model_output)

plt.plot(cor/max(cor))
plt.show()

max_i = np.argmax(cor)
sm = difflib.SequenceMatcher(None, model_output, dataset[max_i : max_i + len(model_output)])
print(sm.ratio(), max_i)
