# pAIno AI-piano

pAIno is a project done by [me](https://github.com/strajdzsha) and [srete](https://github.com/srete) as part of PSI:ML8.

Idea was to create a model that can generate polyphonic music for piano.

# How to run it?

You can find pretrained model [here](https://drive.google.com/drive/folders/13S-rnXaWo_n5oGZHm8_U4Z8Y7B233R20?usp=sharing) and dataset [here](https://github.com/strajdzsha/pAIno---AI-piano/blob/main/dataset.npy). Once you downloaded both files you can run generate_music.py.
Note: pay attention to this block:

```
with torch.no_grad():
    context = loadRandomSong(path="maestro-v3.0.0\\2004\\", length=150).squeeze().tolist() # remi representation of prompt
    context = [train_dataset.stoi[x] for x in context] # going form remi representation to something model would train on
    x = torch.tensor(context, dtype=torch.long)[None,...].to(trainer.device)
    y = model.generate(x, 1000, temperature=1.0, do_sample=True, top_k=10)[0]
    converted_y = np.array([train_dataset.itos[c.item()] for c in y]) # reverting back to remi representation
```
You can specify your own context in any way you want, and set the parameters for temperature, max_tokens and top_k.
If something doesn't work, you may want to check [requirements](https://github.com/strajdzsha/pAIno---AI-piano/edit/main/README.md#requirements) for library versions.

# Dataset preparation
We used [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset which contains ~1k midi files, each of length 3-5min.
For preprocessing we used [Natooz's](https://github.com/Natooz) library miditok, whose documentation can be found [here](https://github.com/Natooz/MidiTok). We used REMI representation of notes, which converts midi file to 1D numpy array that can be directly fed into model. <br/>

<p align="center">
  <img src="https://github.com/strajdzsha/pAIno---AI-piano/blob/main/src/remi.png" />
</p>

# Model
For the model we used [karpathy's](https://github.com/karpathy) excellent gpt2 implementation called [minGPT](https://github.com/karpathy/minGPT). It is easy to implement and even to understand the code for the model.<br/>
We got best results for model gpt2 (117M parameters), block_size = 512 and learning_rate = 1e4, which was trained for 50k iterations. You can find it on before mentioned [location](https://drive.google.com/drive/folders/13S-rnXaWo_n5oGZHm8_U4Z8Y7B233R20?usp=sharing). 

# Requirements
Library versions we installed:

miditok v1.2.7<br/>
mingpt v0.0.1 <br/>
torch v1.12.0 <br/>
numpy v1.23.1 <br/>

# TODO

- A lot of work needs to be done on piano-test-v1.py
- Clean up code for generate_music.py, unnecessary loading of dataset slowes down generating music.
- Train model on different, simpler, datasets
- Upgrade model so it can proccess multi instrument tracks
