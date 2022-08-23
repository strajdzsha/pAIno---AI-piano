import time
from statistics import mode
import numpy as np
import difflib
import matplotlib.pyplot as plt

'''Checks for overfit in model outputs; you wont really need this...'''


dataset = np.load('dataset.npy')
model_output = np.load('evaluation/output_mini_40.npy')
song = np.array([1, 204, 53, 105, 124, 207, 37, 101, 126, 208, 53, 104, 130, 210, 41, 104, 125, 213, 44])

cor = np.correlate(dataset, model_output)

plt.plot(cor/max(cor))
plt.show()

max_i = np.argmax(cor)
sm = difflib.SequenceMatcher(None, model_output, dataset[max_i : max_i + len(model_output)])
print(sm.ratio(), max_i)
