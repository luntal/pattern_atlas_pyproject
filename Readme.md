# Pattern Atlas – Python Development Project

A Python-based Variational Autoencoder (VAE) for generating and transforming rhythmic MIDI patterns.  
This project is intended for both experimentation and development of model architectures, input features, and latent space exploration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development Notes](#development-notes)
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

## Project Structure

```text
my-project/
├── src/                   # Python source code
├── data/                  # Preprocessed tokenized data (.npz)
├── models/                # Trained VAE models (.pt, .onnx, etc.)
├── outputs/               # Generated Sequences, reconstructions (.pkl)
├── tests/                 # Unit tests
├── docs/                  # Development notes (internal)
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore
└── pyproject.toml
```

---

## Installation
1. Clone the repository:
2. Install dependencies:
3. Download the groove midi dataset:

---
## Usage
1. Training via train.py
2. sampling via sample.py
3. hyperparameter tuning in hparams.py

---
## Development Notes
- Internal development notes are maintained in DEV_NOTES.md￼
- Includes:
  - Model architecture changes
  - Input feature experiments
  - Known issues / limitations
  - TODOs for current commit/version

---
## Dataset
- Original dataset: Groove Midi Dataset
- Preprocessed dataset can be created via data.py

---
## License
MIT License – See License file for details