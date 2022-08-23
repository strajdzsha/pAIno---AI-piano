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
If something doesn't work, you may want to check requirements for library versions.

# Dataset preparation


# Model

# Requirements

# TODO
