import numpy as np
import torch
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize, Encoder, Decoder
from modules.interpolation import interpolate_latent_vectors_spherical
from modules.utilities import seq2mat
from modules.midi_processor import MidiProcessor

from types import SimpleNamespace

# Load the saved data
checkpoint = torch.load('model96_3.pt')
# checkpoint = torch.load('model.pt')
model_state_dict = checkpoint['model_state_dict']
hparams_dict = checkpoint['hparams']

# Recreate the hparams object
hparams = SimpleNamespace(**hparams_dict)


model = TransformerVAE(hparams)
model.load_state_dict(model_state_dict)
model.eval()

print('--' * 10)
print('model config: ')
print('model dimensions: ', hparams.d_model)
print('n_layers: ', hparams.n_layers)
print('n_heads: ', hparams.n_heads)
print('--' * 10)

###########################
# Load Midi Data
###########################
abs_path = '/Users/alexanderlunt/Music/Ableton/Projects/Projektrealisation/Drums-Patterns Project/Midi-Clips'
processor = MidiProcessor(hparams)
processor.extract(abs_path, 0, 0)
# processor.visualize(playback=False)
data = processor.convert_to_numpy()



# position = 250
# data = np.load('dataset.npz')['tensor']
# print(data['tensor'])
print('data shape: ', data.shape)
maxi = data.shape[0]
# maxi = hparams.n_samples + 200
# print('using: ', maxi, ' samples')
data_tensors = torch.tensor(data[0], dtype=torch.long)
data_tensors[data_tensors >= 60] = 59
data_tensors = data_tensors[data_tensors != 0].view(1, -1) # remove zeros and reshape
print(data_tensors.shape)
# exit()
onnx_program = torch.onnx.export(model, data_tensors, dynamo=True)
# onnx_program = torch.onnx.export(model, data_tensors, "model.onnx",
#                   opset_version=17,
#                   input_names=['tokens'],
#                   output_names=['logits', 'mu', 'logvar'])
onnx_program.save("model.onnx")

import onnx

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)