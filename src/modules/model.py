import torch
import torch.nn as nn
# from vae_rnn_09.hparams import hparams
from torchtune.modules import RotaryPositionalEmbeddings
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from types import SimpleNamespace
import onnx
import onnxruntime
import math
import csv
from datetime import datetime
import os

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # returns KL per sample (sum over latent dims)
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


class Encoder(nn.Module):
    def __init__(self, _hparams: SimpleNamespace):
        super().__init__()
        self.h_dim = _hparams.d_model // _hparams.n_heads
        self.n_heads = int(_hparams.n_heads)
        self.d_model = int(_hparams.d_model)
        self.n_layers = int(_hparams.n_layers)
        self.latent_dim = _hparams.latent_dim
        self.n_tokens = _hparams.vocab_size
        self.max_seq_len = int(_hparams.max_seq_len)

        beat_emb_dim = _hparams.total_steps // 2
        self.embedding = nn.Embedding(_hparams.vocab_size, self.d_model)
        self.beat_embedding = nn.Embedding(_hparams.total_steps, beat_emb_dim)
        self.projection = nn.Linear(self.d_model+beat_emb_dim, self.d_model)
        # Rotary Positional Embedding
        # self.positional_embedding = RotaryPositionalEmbeddings(self.h_dim)
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            batch_first=True,
            dim_feedforward=_hparams.ff_dim,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers, enable_nested_tensor=False)

        # Learnable [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # Output Projection
        self.out = nn.Linear(self.d_model, self.d_model)
        self.fc_mu = nn.Linear(self.d_model, _hparams.latent_dim)
        self.fc_logvar = nn.Linear(self.d_model, _hparams.latent_dim)

    def scalars_to_normalized_tensor(self, v: torch.Tensor, total_steps: float, length: int):
        # v: [B] (float or int)
        v = v.to(torch.float32)
        norm = (v / max(1.0, float(total_steps))).clamp(0.0, 1.0)  # [B]
        return norm.unsqueeze(1).expand(-1, length)  # [B, length]

    def forward(self, x, beat_tokens):
        src_key_padding_mask = (x == 0)
        x = self.embedding(x)

        # beat tokens as concatenated vector, then projected to d_model
        beat_tokens = self.beat_embedding(beat_tokens)
        x = torch.cat((x, beat_tokens), dim=-1)
        x = self.projection(x)

        # # beat tokens as added vector
        # x = x + beat_tokens
        # Create tgt_key_padding_mask where tokens equal to 0 are masked

        # print(src_key_padding_mask)

        pos_emb = self.positional_embedding(torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
        x += pos_emb

        ## rotary positional embedding
        # x = x.view(x.shape[0], x.shape[1], self.n_heads, self.h_dim)
        # x = self.positional_embedding(x)
        # x = x.view(x.shape[0], x.shape[1], self.d_model)


        # Expand cls token for batch
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [batch, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)  # Result: (batch, seq_len + 1, d_model)
        # print('x shape                   : ', x.shape)

        # Add column of False to the mask for the CLS token
        cls_padding = torch.zeros(src_key_padding_mask.size(0), 1, dtype=torch.bool, device=src_key_padding_mask.device)
        src_key_padding_mask = torch.cat((cls_padding, src_key_padding_mask), dim=1)
        # print('src_key_padding_mask shape: ', src_key_padding_mask.shape)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Extract CLS representation
        x = x[:, 0, :]  # [batch, d_model]

        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        # x = self.out(cls_repr)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar



# ------------------------- Transformer Decoder -------------------------
class Decoder(nn.Module):
    def __init__(self, _hparams: SimpleNamespace):
        super().__init__()
        self.vocab_size = _hparams.vocab_size
        self.emb_size = int(_hparams.d_model)
        self.sos_token = _hparams.sos_token
        self.eos_token = _hparams.eos_token
        self.pad_token = _hparams.pad_token
        self.step_token = _hparams.step_token
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=_hparams.pad_token)
        beat_emb_dim = self.emb_size//2
        self.beat_embedding = nn.Embedding(_hparams.total_steps, beat_emb_dim)
        self.projection = nn.Linear(self.emb_size + beat_emb_dim, self.emb_size)
        self.pos_embedding = nn.Embedding(_hparams.max_seq_len, self.emb_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_size, nhead=_hparams.n_heads, dim_feedforward=_hparams.ff_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=_hparams.n_layers)

        self.z_to_mem = nn.Linear(_hparams.latent_dim, self.emb_size)
        self.fc_out = nn.Linear(self.emb_size, self.vocab_size)
        self.max_seq_len = _hparams.max_seq_len

    def forward(self, tgt_tokens, z, beat_tokens):
        B, T = tgt_tokens.size()
        # if T > self.max_seq_len:
        #     raise ValueError(f"T={T} > max_seq_len={self.max_seq_len}")

        emb = self.token_embedding(tgt_tokens)
        beat_emb = self.beat_embedding(beat_tokens)
        emb = self.projection(torch.cat((emb, beat_emb), dim=-1))
        pos_emb = self.pos_embedding(torch.arange(T, device=tgt_tokens.device)).unsqueeze(0)
        emb += pos_emb
        # emb = emb + pos_emb


        memory = self.z_to_mem(z).unsqueeze(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)
        tgt_key_padding_mask = (tgt_tokens == self.pad_token).float()
        out = self.transformer_decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def sample(self, z, max_len, temperature=1.0, sample=True, device=None):
        z = z.to(device)
        B = z.size(0)
        device = device or next(self.parameters()).device
        cur_tokens = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)
        beat_tokens = torch.zeros((B, 1), dtype=torch.long, device=device)
        memory = self.z_to_mem(z).unsqueeze(1)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            T = cur_tokens.size(1)
            if T > self.max_seq_len or finished.all():
                break
            emb = self.token_embedding(cur_tokens)
            beat_emb = self.beat_embedding(beat_tokens)
            emb = self.projection(torch.cat((emb, beat_emb), dim=-1))
            pos_emb = self.pos_embedding(torch.arange(T, device=device)).unsqueeze(0)
            emb = emb + pos_emb

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            tgt_key_padding_mask = (cur_tokens == self.pad_token).float()
            out = self.transformer_decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            logits = self.fc_out(out)
            last_logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = F.softmax(last_logits, dim=-1)
            if sample:
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)


            cur_tokens = torch.cat([cur_tokens, next_tok], dim=1)

            # #make sure beat_tokens[0] = 0, and afterwards starting with 1
            is_step = (cur_tokens == self.step_token).to(torch.long)  # [B, T] in {0,1}
            beat_tokens = torch.cumsum(is_step, dim=1)  # [B, T]
            beat_tokens += 1
            beat_tokens[:, 0] = 0

            finished |= (next_tok.squeeze(1) == self.eos_token)
        return cur_tokens[:, 1:]


# ------------------------- VAE wrapper -------------------------
class TransformerVAE(nn.Module):
    def __init__(self, _hparams: SimpleNamespace):
        super().__init__()
        self.encoder = Encoder(_hparams)
        self.decoder = Decoder(_hparams)
        self.latent_dim = _hparams.latent_dim
        self.step_token = _hparams.step_token

    def forward(self, tokens):
        B, T = tokens.size()
        sos = torch.full((B, 1), self.decoder.sos_token, dtype=torch.long, device=tokens.device)
        decoder_input = torch.cat([sos, tokens[:, :-1]], dim=1)
        # mask the input eos token
        decoder_input[decoder_input == 2] = 0
        # decoder_input = torch.where(decoder_input == 2, torch.zeros_like(decoder_input), decoder_input)
        beat_tokens_enc = self.make_beat_tokens(tokens, step_token=int(self.step_token))

        beat0 = torch.zeros((B, 1), dtype=torch.long, device=tokens.device)
        beat_tokens_dec = torch.cat([beat0, beat_tokens_enc[:, :-1]], dim=1)

        # print('enc input shape: ', tokens.shape)
        mu, logvar = self.encoder(tokens, beat_tokens_enc)
        z = reparameterize(mu, logvar)
        logits = self.decoder(decoder_input, z, beat_tokens_dec)
        return logits, mu, logvar

    def make_beat_tokens(self, tokens: torch.Tensor, step_token: int) -> torch.Tensor:
        # tokens: [B, T] (long)
        is_step = (tokens == step_token).to(torch.long)  # [B, T] in {0,1}
        beat_tokens = torch.cumsum(is_step, dim=1)  # [B, T]
        beat_tokens += 1
        beat_tokens[:,0] = 0
        # padding with zeros
        pad_mask = (tokens == self.decoder.pad_token)
        beat_tokens = beat_tokens.masked_fill(pad_mask, 0)
        return beat_tokens

    @torch.no_grad()
    def sample(self, z=None, max_len=128, temperature=1.0, sample=True, device=None):
        device = device or next(self.parameters()).device
        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)
        return self.decoder.sample(z.to(device), max_len=max_len, temperature=temperature, sample=sample, device=device)


# ------------------------- Loss & Training helpers -------------------------
def vae_losss(logits, targets, mu, logvar, beta=1.0, free_bits=0.0, pad_token=0):
    B = targets.size(0)
    T = targets.size(1)
    recon = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token, reduction='sum') / (B * T)
    # kl = kl_divergence(mu, logvar)
    # kl = kl_divergence(mu, logvar)# .mean() / targets.size(1)  # Normalize by seq_len
    # kl divergence:
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)

    if free_bits > 0:
        min_kl = free_bits # * mu.size(-1)
        # kl = torch.clamp(kl / mu.size(-1), min=min_kl).mean()
        kl = torch.clamp(kl, min=min_kl).mean()
    else:
        # kl = kl.mean() / mu.size(-1)
        kl = kl.mean()
    kl = beta * kl
    loss = recon + kl


    return loss, recon, kl

def vae_loss(logits, targets, mu, logvar, beta=1.0, free_bits=0.0, pad_token=0):
    B, T = targets.size()
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    mask = (targets_flat != pad_token)
    token_count = mask.sum().clamp(min=1).to(logits.device, dtype=logits.dtype)
    ce_sum = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token, reduction="sum")
    # loss per token
    recon = ce_sum / token_count

    # kl per sample
    kl_per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )

    # number of valid tokens per sample
    tokens_per_sample = (targets != pad_token).sum(dim=1).clamp(min=1).to(logits.dtype)  # [B]

    # normalize KL per token
    kl_per_token = kl_per_sample / tokens_per_sample  # [B]
    if free_bits > 0:
        kl_per_token = torch.clamp(kl_per_token, min=free_bits)
    kl = kl_per_token.mean()
    kl = beta * kl
    total_loss = recon + kl

    # info = (
    #     f"CE/token: {recon.item():.4f} "
    #     f"PPL: {torch.exp(torch.clamp(recon, max=20)).item():.4f} "
    # )
    perplexity = torch.exp(torch.clamp(recon, max=20)).item()

    return total_loss, recon, kl, perplexity, token_count

def train_vae(
        model: TransformerVAE, dataloader: DataLoader, _hparams: SimpleNamespace,
        optimizer, device, model_name='../models/model.pt', annealing_style='sigmoid', scheduler=None, step_start=0):
    epochs = _hparams.n_epochs
    # annealing_steps = _hparams.annealing_steps
    # calculate annealing steps based on number of samples, should be approx. 50%
    annealing_steps = len(dataloader) * (epochs // 2)
    beta = _hparams.beta
    free_bits = _hparams.free_bits
    loss_history = []

    # Initialize CSV logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'docs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.csv')

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch_size', 'total_loss', 'reconstruction_loss', 'kl_loss',
            'perplexity', 'avg_tokens_per_sequence', 'learning_rate', 'beta',
            'annealing_factor', 'epoch_time_seconds', 'elapsed_time_seconds',
            'remaining_time_seconds', 'saved_cache'
        ])

    prev_loss = None
    model.to(device)
    # transfer optimizer state tensors to device:
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)



    total_steps = 0 + _hparams.global_step

    train_start_time = time.time()
    prev_epoch_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = {'loss': 0.0, 'rec': 0.0, 'kl': 0.0, 'n_tokens': 0.0}
        batch_iteration = 0
        prev_time = time.time()
        perplexity = 1000
        for batch in dataloader:
            # Compute annealing factor
            if total_steps < annealing_steps:
                # Linear schedule from 0 to 1
                if annealing_style == 'linear':
                    annealing_factor = total_steps / annealing_steps

                # Sigmoid schedule - smoother transition
                # annealing_factor = 1.0 / (1.0 + np.exp(-10 * (total_steps / annealing_steps - 0.5)))
                if annealing_style == 'sigmoid':
                    # MuseMorphose-inspired Sigmoid Annealing
                    # This starts slow, then rises quickly around the midpoint of annealing_steps
                    x_prop = (total_steps / annealing_steps)
                    annealing_factor = 1.0 / (1.0 + np.exp(-10 * (x_prop - 0.5)))

            else:
                annealing_factor = 1.0
            total_steps += 1
            weighted_beta = beta * annealing_factor
            time_elapsed = time.time() - prev_time
            prev_time = time.time()
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(batch)
            loss, rec, kl, perplexity, token_count = vae_loss(logits, batch, mu, logvar, beta=weighted_beta, free_bits=free_bits, pad_token=_hparams.pad_token)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f'\rEpoch {epoch:03d}/{epochs} | Step {batch_iteration + 1:2d}/{len(dataloader)} | step_time: {time_elapsed:.2f}s', end='')
            batch_iteration += 1
            if scheduler is not None:
                scheduler.step()
            running['loss'] += loss.detach()
            running['rec'] += rec.detach()
            running['kl'] += kl.detach()
            running['n_tokens'] += token_count.detach()
        n_batches = len(dataloader)
        print()

        now = time.time()
        epoch_time = now - prev_epoch_time
        prev_epoch_time = now
        current_lr = optimizer.param_groups[0]['lr']
        _hparams.global_step = total_steps
        _hparams.global_epoch = total_steps // n_batches
        _hparams.current_lr = current_lr
        loss_history.append(running['loss']/n_batches)
        # if save_dir is not None:
        # torch.save({'epoch': epoch, 'model_state': model.state_dict()}, 'model_cache.pt')
        save_cache = False
        if prev_loss is not None and prev_loss > running['loss']/n_batches:
            save_model(_hparams, model, '../models/model_cache.pt', scheduler, optimizer, loss_history)
            save_cache = True
        # print('previous loss: ', prev_loss, ' , current loss: ', running['loss']/n, '')
        prev_loss = min(running['loss']/n_batches, prev_loss) if prev_loss is not None else running['loss']/n_batches

        print(f"Loss {running['loss']/n_batches:.4f} | rec {running['rec']/n_batches:.4f} | kl {running['kl']/n_batches:.4f} | Epoch_time: {epoch_time:.2f}s | saved cache: {save_cache} | lr: {current_lr:.10f}")
        print(f'Model Perplexity: {perplexity} | Avg tokens per sequence: {running["n_tokens"]/_hparams.n_samples:.2f}')


        # print time elapsed and remaining
        elapsed_time = time.time() - train_start_time
        e_hours = elapsed_time // 3600
        e_minutes = (elapsed_time % 3600) // 60
        e_seconds = int(elapsed_time % 60)
        remaining_time = (elapsed_time/epoch) * (epochs - epoch)
        r_hours = remaining_time // 3600
        r_minutes = (remaining_time % 3600) // 60
        r_seconds = int(remaining_time % 60)
        print(f"Elapsed: {e_hours:.2f}h {e_minutes}min {e_seconds}s | Remaining: {r_hours}h {r_minutes}m {r_seconds}s")

        # Log metrics to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                _hparams.batch_size,
                f"{(running['loss']/n_batches).item():.6f}",
                f"{(running['rec']/n_batches).item():.6f}",
                f"{(running['kl']/n_batches).item():.6f}",
                f"{perplexity:.4f}",
                f"{running['n_tokens']/_hparams.n_samples:.2f}",
                f"{current_lr:.10f}",
                f"{weighted_beta:.6f}",
                f"{annealing_factor:.6f}",
                f"{epoch_time:.2f}",
                f"{elapsed_time:.2f}",
                f"{remaining_time:.2f}",
                save_cache
            ])

        print('-' * 100)
    # torch.save(model.state_dict(), model_name)

    save_model(_hparams, model, model_name, scheduler, optimizer, loss_history)


    print(f"Training took {(time.time() - train_start_time)/60:.2f} minutes")
    print(f'model saved to {model_name}')
    print(f'training log saved to {log_file}')

def save_model(_hparams, model, model_name='../models/model.pt', scheduler=None, optimizer=None, loss_history=None):
    _hparams.scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
    _hparams.optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
    if loss_history is not None:
        _hparams.loss_history = loss_history

    save_dict = {
        'model_state_dict': model.state_dict(),
        'hparams': vars(_hparams)  # Convert SimpleNamespace to dict
    }

    torch.save(save_dict, model_name)


# ------------------------- ONNX Wrapper ---------------------------------
class ONNXWrapper(nn.Module):
    def __init__(self, _hparams: SimpleNamespace, encoder_path='../models/encoder.onnx', decoder_path='../models/decoder.onnx'):
        super().__init__()
        # Use ONNX Runtime sessions instead of onnx.load
        self.enc_session = onnxruntime.InferenceSession(encoder_path)
        self.dec_session = onnxruntime.InferenceSession(decoder_path)

        self.latent_dim = _hparams.latent_dim
        self.sos_token = _hparams.sos_token
        self.eos_token = _hparams.eos_token
        self.max_seq_len = _hparams.max_seq_len

    def forward(self, tokens):
        if type(tokens) is not np.ndarray:
            # Dynamic axes handle B and T. We just convert to numpy.
            tokens_np = tokens.cpu().numpy().astype(np.int64)
        else:
            tokens_np = tokens.astype(np.int64)
        current_len = tokens_np.shape[1]
        if current_len < self.max_seq_len:
            tokens_np = np.pad(tokens_np, ((0, 0), (0, self.max_seq_len - current_len)), mode='constant',
                               constant_values=0)
        elif current_len > self.max_seq_len:
            tokens_np = tokens_np[:, :self.max_seq_len]

        # Run Encoder: expects [B, T]
        outputs = self.enc_session.run(None, {'input_tokens': tokens_np})
        mu = torch.from_numpy(outputs[0])
        logvar = torch.from_numpy(outputs[1])

        z = reparameterize(mu, logvar)

        # Teacher Forcing: Shift tokens and add SOS
        B, T = tokens_np.shape
        sos = np.full((B, 1), self.sos_token, dtype=np.int64)
        dec_input = np.concatenate([sos, tokens_np[:, :-1]], axis=1)
        dec_input[dec_input == 2] = 0

        dec_input_np = dec_input.astype(np.int64)
        z_np = z.cpu().numpy().astype(np.float32)

        # Decoder session runs with actual sequence length T
        logits_np = self.dec_session.run(None, {'tgt_tokens': dec_input_np, 'latent_z': z_np})[0]

        logits = torch.from_numpy(logits_np)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, z: torch.Tensor = None, max_len=128, temperature=1.0, sample=True, device=None):
        device = device or 'cpu'
        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)

        z_np = z.cpu().numpy().astype(np.float32)
        B = z_np.shape[0]
        all_results = []
        cur_tokens_np = np.zeros((B, self.max_seq_len), dtype=np.int64)
        cur_tokens_np[:, 0] = self.sos_token
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(self.max_seq_len-1):
            # Autoregressive: T grows from 1 to max_len, BUT: shape has to maintain (B, max_seq_len)
            # always padded with eos_token!!!


            # session.run accepts the dynamic T dimension
            logits_np = self.dec_session.run(None, {'tgt_tokens': cur_tokens_np, 'latent_z': z_np})[0]
            logits = torch.from_numpy(logits_np)

            last_logits = logits[:, t, :] / max(1e-8, temperature)
            probs = F.softmax(last_logits, dim=-1)
            if sample:
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)
            if next_tok.shape[1] != 1:
                print('Warning next tok has wrong shape: ', next_tok.shape, '')
                exit()
            cur_tokens_np[:, t + 1] = next_tok.squeeze(1).numpy()
            finished |= (next_tok.squeeze(1) == self.eos_token)

        return cur_tokens_np[:, 1:]
