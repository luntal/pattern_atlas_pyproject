## VAE Transformer
- with beat embedding


### model features
- spherical interpolation!
- microtiming, 24 pitches, velocity
- model.pt
- mixed dataset
  - rhythm and melody
  - quantized and original and in between
- beat embedding
  - Option A: summation of beat embeddings and midi event vectors
  - Option B: concatenation of beat embeddings and midi event vectors, then linear projection

### Todo
- integrate note off
- integrate micro timing ✅

### Changes
- normalization of kl
- setting free bits to 0.25
- beta to 0.05
- annealing-schedule changed to sigmoid

### technical infos
- For each latent dimension:
  - mu is the center / average value
  - std controls how much random spread that dimension has
  - eps * std is the random deviation
  - mu + ... shifts that random deviation around the mean
- if you think of the latent space as a cloud:
  - mu = the center of the cloud
  - std = the cloud radius/spread per dimension
  - reparameterization = pick a random point near that center

### JS Convertion
- load onnx model
- load midi data
- transform midi to tokens
- load tokens to onnx encoder and generate x amount of latent vectors
- reparameterization trick
- interpolation (brf)
- inference: sample from latent space vectors
- transform tokens to midi and to matrices

functions
- load midi from folder
- transform midi to matrices 
- transform tokens to midi

### git
- hard reset:
  - git fetch origin
  - git checkout dev (check that your are in the correct branch!)
  - git reset --hard origin/dev

---
### training

- [text](../models/exp/model_2026-04-20_22-37-21.pt)
- overfitting with latnet to memory expansion
- 