import torch
from modules.model import ONNXWrapper
from hparams import hparams
import numpy as np
model = ONNXWrapper(hparams)
tokens = np.random.randint(3, 59, (1,100), dtype=np.int64)
# tokens = np.pad(tokens, ((0, 0), (0, 156)), mode='constant', constant_values=0)
logits, mu, logvar = model(tokens)
print(f'logits shape: {logits.shape} (B, T, V), type: {logits.dtype}')
print(f'mu shape: {mu.shape} (B, V)')
print(f'logvar shape: {logvar.shape} (B, V)')

B = 1
hparams = hparams
z = torch.randn(B, hparams.latent_dim, dtype=torch.float32)
prediction = model.sample(z)
print(prediction.shape)
