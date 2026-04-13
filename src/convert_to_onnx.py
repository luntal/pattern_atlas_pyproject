import numpy as np
import torch
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize, Encoder, Decoder
from modules.interpolation import interpolate_latent_vectors_spherical
from modules.utilities import seq2mat
from modules.midi_processor import MidiProcessor
import onnx
from types import SimpleNamespace
import glob
import os
from modules.branch_paths import latest_in_branch, safe_branch_name, get_current_branch

# Find the latest model in branch-specific models directory
branch_name = safe_branch_name(get_current_branch())
latest_model_path = latest_in_branch('models', 'model_*.pt')
if latest_model_path is None:
    raise FileNotFoundError(f"No model files found in models/{branch_name}/")
latest_model = str(latest_model_path)
print(f'Loading model: {latest_model}')
# model_name = os.path.basename(latest_model)
# change the model name to have onnx suffix
onnx_encoder_name = latest_model.replace('.pt', '_encoder.onnx')
onnx_decoder_name = latest_model.replace('.pt', '_decoder.onnx')

checkpoint = torch.load(latest_model, map_location='cpu')
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

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters in model: {total_params / 1e6:.2f}M')
print('--' * 10)


# 2. Export Encoder (only calls Encoder.forward)
dummy_enc_input = torch.zeros((1, hparams.max_seq_len), dtype=torch.long)
torch.onnx.export(
    model.encoder,
    (dummy_enc_input, dummy_enc_input),
    onnx_encoder_name,
    input_names=['input_tokens', 'beat_tokens'],
    output_names=['mu', 'logvar'],
    dynamic_axes={
        'input_tokens': {0: 'batch_size', 1: 'seq_len'},
        'beat_tokens': {0: 'batch_size', 1: 'seq_len'}
    },
    opset_version=14,
)

# 3. Export Decoder (only calls Decoder.forward)
# Note: This exports the TEACHER FORCING path (one step or parallel training path)
# which is what you usually want for a separate ONNX decoder.
dummy_dec_tokens = torch.zeros((1, hparams.max_seq_len), dtype=torch.long) # Current sequence
dummy_z = torch.randn(1, hparams.latent_dim) # Latent vector
torch.onnx.export(
    model.decoder,
    (dummy_dec_tokens, dummy_z, dummy_dec_tokens),
    onnx_decoder_name,
    input_names=['tgt_tokens', 'latent_z', 'beat_tokens'],
    output_names=['logits'],
    dynamic_axes={
        'tgt_tokens': {0: 'batch_size', 1: 'seq_len'},
        'latent_z': {0: 'batch_size'},
        'beat_tokens': {0: 'batch_size', 1: 'seq_len'}
    },
    opset_version=14,
)

print("Modules exported. sample() methods were naturally excluded.")


# test load
# onnx_model = onnx.load(onnx_encoder_name)
# onnx.checker.check_model(onnx_model)