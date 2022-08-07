#%%
from statistics import mode
import numpy as np
import difflib

def getRatio(dataset, model_output, shift):
    ratio = []
    i = 0
    while(True):
        sm = difflib.SequenceMatcher(None, model_output, dataset[i : i+len(model_output)])
        ratio.append(sm.ratio())

        i += shift
        if (i + len(model_output) >= len(dataset) - 1): break
        #if(i%10000 == 0): print(i)
    
    return ratio

END_TOKEN = 420

dataset = np.load('D:/PSIML/code/pAIno---AI-piano/Dataset_maestro_with_chords.npy')
model_output = np.load('D:/PSIML/code/pAIno---AI-piano/output_mini_40.npy')
song = np.array([1, 204, 53, 105, 124, 207, 37, 101, 126, 208, 53, 104, 130, 210, 41, 104, 125, 213, 44])
model_output = dataset[500:1500]

#%%
ratio = getRatio(dataset, model_output, shift=len(model_output)//2)

#%%
print(max(ratio))
# %%
