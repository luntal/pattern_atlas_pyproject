# Pattern VAE – Python Development Project

A Python-based Variational Autoencoder (VAE) for generating and transforming rhythmic MIDI patterns.  
This project is intended for both experimentation and development of model architectures, input features, and latent space exploration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [License](#license)

---

## Project Overview

This project implements a VAE to encode MIDI token sequences into latent vectors and decode them back into new rhythmic patterns.  
The model supports:

- Pitch, velocity, micro-timing and step token embeddings
- Latent space interpolation
- Autoregressive decoding

---

## Features

- Encode MIDI sequences into a latent representation
- Decode latent vectors into new sequences
- Grid-based latent space visualization (via Jitter or CLI tools)
- Support for dataset augmentation (pitch shift, timing jitter)
- Flexible architecture for experimenting with VAE parameters


---

## Installation
1. Clone the repository:
2. Install dependencies:
3. Download the groove midi dataset:

---
## Usage
1. Training via train.py
2. sampling via inference.py
3. hyperparameter tuning in hparams.py

---
## Dataset
- Original dataset: Groove Midi Dataset
- Preprocessed dataset can be created via data.py

---
## License
MIT License – See License file for details