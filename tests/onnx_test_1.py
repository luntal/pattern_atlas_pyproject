import onnxruntime
from hparams import hparams
import numpy as np

enc_session = onnxruntime.InferenceSession('encoder.onnx')
dec_session = onnxruntime.InferenceSession('decoder.onnx')

encoder_input = np.random.randint(3, 59, (1,100), dtype=np.int64)
encoder_input = np.pad(encoder_input, ((0, 0), (0, 156)), mode='constant', constant_values=0)

encoder_output = enc_session.run(None, {'input_tokens': encoder_input})
mu, logvar = encoder_output
print(f'mu: {mu.shape}, logvar: {logvar.shape}')

B = 5
decoder_tokens = np.random.randint(3, 59, (B,100), dtype=np.int64)
decoder_tokens = np.insert(decoder_tokens, 0, 1, axis=1)
decoder_tokens = np.pad(decoder_tokens, ((0, 0), (0, 155)), mode='constant', constant_values=0)
z = np.random.randn(B, hparams.latent_dim).astype(np.float32)
decoder_output = dec_session.run(None, {'tgt_tokens': decoder_tokens, 'latent_z': z})
logits = decoder_output[0]
print(f'logits shape: {logits.shape} (B, T, V), type: {type(logits)}')
