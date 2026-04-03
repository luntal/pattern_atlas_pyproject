from modules.model import TransformerVAE
import torch
from hparams import hparams

t = TransformerVAE(hparams)
tensor = torch.tensor([[3,2,5,6,3,4,5,6,3,7,8,9,10]])
step_token = 3
beat_tokens = t.make_beat_tokens(tensor, step_token)
print(tensor)
print(beat_tokens)