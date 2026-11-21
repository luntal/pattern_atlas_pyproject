import torch
import torch.nn as nn
# from torchtune.modules import RotaryPositionalEmbeddings
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from types import SimpleNamespace
import onnx
import onnxruntime
import math
import csv
from modules.branch_paths import branch_dir, safe_branch_name, get_current_branch
from datetime import datetime
import os
import shutil
from zoneinfo import ZoneInfo

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # returns KL per sample (sum over latent dims)
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


N_FRAME_FIELDS = 4
N_SPECIAL_IDS = 3  # 0 = PAD, 1 = SOS, 2 = EOS in every field; raw values start at this offset


def field_vocab_sizes_of(_hparams: SimpleNamespace) -> list:
    """Vocabulary size per frame field (step, pitch, velocity, deviation) incl. the 3 special ids."""
    sizes = getattr(_hparams, 'field_vocab_sizes', None)
    if sizes:
        return [int(s) for s in sizes]
    return [int(_hparams.width) + N_SPECIAL_IDS, int(_hparams.n_pitch) + N_SPECIAL_IDS,
            int(_hparams.n_vel) + N_SPECIAL_IDS, int(_hparams.n_timing) + N_SPECIAL_IDS]


def field_slices_of(field_vocab_sizes) -> list:
    """(start, end) of every field inside the concatenated output logits."""
    slices, start = [], 0
    for size in field_vocab_sizes:
        slices.append((start, start + int(size)))
        start += int(size)
    return slices


class FieldEmbedding(nn.Module):
    """Sum of one embedding per frame field, scaled by 1/sqrt(n_fields). frames: [..., 4] -> [..., d_model].

    The absolute position inside the 2-bar window is carried by the step field.
    """

    def __init__(self, field_vocab_sizes, d_model: int, pad_token: int = 0):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(int(size), int(d_model), padding_idx=pad_token) for size in field_vocab_sizes
        )
        self.scale = 1.0 / math.sqrt(len(self.embeddings))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        out = self.embeddings[0](frames[..., 0])
        for i in range(1, len(self.embeddings)):
            out = out + self.embeddings[i](frames[..., i])
        return out * self.scale


class Encoder(nn.Module):
    def __init__(self, _hparams: SimpleNamespace):
        super().__init__()
        self.h_dim = _hparams.d_model // _hparams.n_heads
        self.n_heads = int(_hparams.n_heads)
        self.d_model = int(_hparams.d_model)
        self.n_layers = int(_hparams.n_layers)
        self.latent_dim = _hparams.latent_dim
        self.max_seq_len = int(_hparams.max_seq_len)
        self.pad_token = int(getattr(_hparams, 'pad_token', 0))
        self.field_vocab_sizes = field_vocab_sizes_of(_hparams)

        self.embedding = FieldEmbedding(self.field_vocab_sizes, self.d_model, self.pad_token)
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
        self.fc_mu = nn.Linear(self.d_model, _hparams.latent_dim)
        self.fc_logvar = nn.Linear(self.d_model, _hparams.latent_dim)

    def forward(self, frames):
        """frames: [B, T, 4] (step, pitch, velocity, deviation) -> mu, logvar [B, latent_dim]."""
        # PAD frames (step field == PAD) are ignored by the attention.
        src_key_padding_mask = (frames[..., 0] == self.pad_token)
        x = self.embedding(frames)

        pos_emb = self.positional_embedding(torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
        x = x + pos_emb

        # Expand cls token for batch
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [batch, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)  # Result: (batch, seq_len + 1, d_model)

        # Add column of False to the mask for the CLS token
        cls_padding = torch.zeros(src_key_padding_mask.size(0), 1, dtype=torch.bool, device=src_key_padding_mask.device)
        src_key_padding_mask = torch.cat((cls_padding, src_key_padding_mask), dim=1)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Extract CLS representation
        x = x[:, 0, :]  # [batch, d_model]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar



# ------------------------- Transformer Decoder -------------------------
class Decoder(nn.Module):
    """Autoregressive frame decoder with four parallel output heads (one per field on the same state)."""

    def __init__(self, _hparams: SimpleNamespace):
        super().__init__()
        self.emb_size = int(_hparams.d_model)
        self.sos_token = int(_hparams.sos_token)
        self.eos_token = int(_hparams.eos_token)
        self.pad_token = int(_hparams.pad_token)
        self.field_vocab_sizes = field_vocab_sizes_of(_hparams)
        self.field_slices = field_slices_of(self.field_vocab_sizes)
        self.token_embedding = FieldEmbedding(self.field_vocab_sizes, self.emb_size, self.pad_token)
        self.pos_embedding = nn.Embedding(_hparams.max_seq_len, self.emb_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_size, nhead=_hparams.n_heads, dim_feedforward=_hparams.ff_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=_hparams.n_layers)

        self.n_memory_tokens = getattr(_hparams, 'n_memory_tokens', 1)
        self.z_to_mem = nn.Linear(_hparams.latent_dim, self.emb_size * self.n_memory_tokens)
        # One linear layer for all heads; the logits are split per field with field_slices.
        self.fc_out = nn.Linear(self.emb_size, sum(self.field_vocab_sizes))
        self.max_seq_len = _hparams.max_seq_len

    def split_logits(self, logits: torch.Tensor) -> list:
        """[..., sum(field_vocab_sizes)] -> list of 4 tensors [..., field_vocab_size]."""
        return [logits[..., s:e] for s, e in self.field_slices]

    def special_frame(self, kind_id: int, batch: int, device) -> torch.Tensor:
        return torch.full((batch, N_FRAME_FIELDS), int(kind_id), dtype=torch.long, device=device)

    def forward(self, tgt_frames, z):
        """tgt_frames: [B, T, 4] decoder input (SOS frame first), z: [B, latent_dim] -> logits [B, T, sum]."""
        B, T = tgt_frames.shape[:2]

        emb = self.token_embedding(tgt_frames)
        pos_emb = self.pos_embedding(torch.arange(T, device=tgt_frames.device)).unsqueeze(0)
        emb = emb + pos_emb

        memory = self.z_to_mem(z).view(B, self.n_memory_tokens, self.emb_size)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_frames.device)
        tgt_key_padding_mask = (tgt_frames[..., 0] == self.pad_token).float()
        out = self.transformer_decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.fc_out(out)
        return logits

    # ------------------------- KV-cache incremental decoding -------------------------
    # The functions below mirror `forward` (post-norm / pre-norm nn.TransformerDecoderLayer) for one new frame
    # at a time. Self-attention keys/values of all previous positions and the cross-attention keys/values of
    # the latent memory are cached, so every step costs O(T) instead of re-running the whole sequence.
    # They are meant for inference (eval mode, no dropout).
    @staticmethod
    def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, n_heads, D // n_heads).transpose(1, 2)

    @torch.no_grad()
    def init_cache(self, z: torch.Tensor) -> dict:
        """Build a decoding cache for a batch of latent vectors z: [B, latent_dim]."""
        B = z.size(0)
        D = self.emb_size
        memory = self.z_to_mem(z).view(B, self.n_memory_tokens, D)
        mem_kv = []
        for layer in self.transformer_decoder.layers:
            mha = layer.multihead_attn
            w, b = mha.in_proj_weight, mha.in_proj_bias
            k = F.linear(memory, w[D:2 * D], None if b is None else b[D:2 * D])
            v = F.linear(memory, w[2 * D:], None if b is None else b[2 * D:])
            mem_kv.append((self._split_heads(k, mha.num_heads), self._split_heads(v, mha.num_heads)))
        return {
            'mem_kv': mem_kv,
            'self_kv': [None] * len(mem_kv),
            'pad_bias': torch.zeros(B, 1, 1, 0, dtype=memory.dtype, device=memory.device),
            'pos': 0,
        }

    @torch.no_grad()
    def decode_step(self, frame: torch.Tensor, cache: dict) -> torch.Tensor:
        """Feed one frame per sequence (frame: [B, 4]) and return the logits of the next frame [B, sum].

        The first call must feed the SOS frame.
        """
        pos = cache['pos']
        if pos >= self.max_seq_len:
            raise ValueError(f"pos={pos} >= max_seq_len={self.max_seq_len}")
        B = frame.size(0)
        D = self.emb_size

        x = self.token_embedding(frame)
        x = (x + self.pos_embedding.weight[pos]).unsqueeze(1)  # [B, 1, D]

        # Same quirk as the training path: the float padding mask is added to the attention scores.
        pad_now = (frame[:, 0] == self.pad_token).to(x.dtype).view(B, 1, 1, 1)
        pad_bias = torch.cat((cache['pad_bias'], pad_now), dim=-1)
        cache['pad_bias'] = pad_bias

        for i, layer in enumerate(self.transformer_decoder.layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            H = sa.num_heads

            def self_attn_block(h):
                q, k, v = F.linear(h, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
                q, k, v = (self._split_heads(t, H) for t in (q, k, v))
                prev = cache['self_kv'][i]
                if prev is not None:
                    k = torch.cat((prev[0], k), dim=2)
                    v = torch.cat((prev[1], v), dim=2)
                cache['self_kv'][i] = (k, v)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_bias)
                return sa.out_proj(o.transpose(1, 2).reshape(B, 1, D))

            def cross_attn_block(h):
                b = ca.in_proj_bias
                q = F.linear(h, ca.in_proj_weight[:D], None if b is None else b[:D])
                q = self._split_heads(q, H)
                k, v = cache['mem_kv'][i]
                o = F.scaled_dot_product_attention(q, k, v)
                return ca.out_proj(o.transpose(1, 2).reshape(B, 1, D))

            def ff_block(h):
                return layer.linear2(layer.activation(layer.linear1(h)))

            if layer.norm_first:
                x = x + self_attn_block(layer.norm1(x))
                x = x + cross_attn_block(layer.norm2(x))
                x = x + ff_block(layer.norm3(x))
            else:
                x = layer.norm1(x + self_attn_block(x))
                x = layer.norm2(x + cross_attn_block(x))
                x = layer.norm3(x + ff_block(x))

        if self.transformer_decoder.norm is not None:
            x = self.transformer_decoder.norm(x)
        cache['pos'] = pos + 1
        return self.fc_out(x[:, 0, :])

    @staticmethod
    def _choose(logits: torch.Tensor, invalid: torch.Tensor, sample: bool, temperature: float) -> torch.Tensor:
        logits = logits.float().masked_fill(invalid, float('-inf'))
        if sample:
            probs = F.softmax(logits / max(1e-8, float(temperature)), dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def pick_frame(self, logits: torch.Tensor, prev_frame: torch.Tensor | None = None, sample: bool = False,
                   temperature: float = 1.0, constrained: bool = True) -> torch.Tensor:
        """Choose the next frame [B, 4] from the parallel head logits [B, sum].

        Fields are chosen in the order step -> pitch -> velocity / deviation. PAD and SOS are never chosen,
        EOS only in the step field (an EOS step gives an EOS frame). With ``constrained`` (default) the step
        never falls below the previous note's step, and at the same step the pitch must rise above the previous
        pitch. If no higher pitch is left, the same step is blocked too (next step or EOS).
        """
        step_l, pitch_l, vel_l, dev_l = self.split_logits(logits)
        B, device = logits.size(0), logits.device
        eos = self.eos_token

        step_idx = torch.arange(step_l.size(-1), device=device)
        pitch_idx = torch.arange(pitch_l.size(-1), device=device)
        step_invalid = ((step_idx < N_SPECIAL_IDS) & (step_idx != eos)).unsqueeze(0).expand(B, -1)
        pitch_invalid = (pitch_idx < N_SPECIAL_IDS).unsqueeze(0).expand(B, -1)

        if constrained and prev_frame is not None:
            prev_frame = prev_frame.to(device=device, dtype=torch.long)
            prev_step, prev_pitch = prev_frame[:, 0], prev_frame[:, 1]
            is_note = (prev_step >= N_SPECIAL_IDS) & (prev_pitch >= N_SPECIAL_IDS)
            no_pitch_left = is_note & (prev_pitch >= pitch_l.size(-1) - 1)
            falls = is_note.unsqueeze(1) & (step_idx.unsqueeze(0) < prev_step.unsqueeze(1)) & (step_idx.unsqueeze(0) != eos)
            same_blocked = no_pitch_left.unsqueeze(1) & (step_idx.unsqueeze(0) == prev_step.unsqueeze(1))
            step_invalid = step_invalid | falls | same_blocked
        step = self._choose(step_l, step_invalid, sample, temperature)

        if constrained and prev_frame is not None:
            same_step = is_note & (step == prev_step)
            not_rising = same_step.unsqueeze(1) & (pitch_idx.unsqueeze(0) <= prev_pitch.unsqueeze(1))
            pitch_invalid = pitch_invalid | not_rising
        pitch = self._choose(pitch_l, pitch_invalid, sample, temperature)

        vel_invalid = (torch.arange(vel_l.size(-1), device=device) < N_SPECIAL_IDS).unsqueeze(0).expand(B, -1)
        dev_invalid = (torch.arange(dev_l.size(-1), device=device) < N_SPECIAL_IDS).unsqueeze(0).expand(B, -1)
        vel = self._choose(vel_l, vel_invalid, sample, temperature)
        dev = self._choose(dev_l, dev_invalid, sample, temperature)

        frame = torch.stack((step, pitch, vel, dev), dim=1)
        is_eos = (step == eos).unsqueeze(1)
        return torch.where(is_eos, torch.full_like(frame, eos), frame)

    @torch.no_grad()
    def generate(self, z, max_len=None, temperature=1.0, sample=False, constrained=True, pad_after_eos=True):
        """Batched autoregressive generation with KV-cache. Returns frames [B, L, 4] (without SOS).

        With `pad_after_eos=True` finished sequences are filled with PAD frames after their EOS frame.
        """
        device = z.device
        B = z.size(0)
        max_len = int(self.max_seq_len if max_len is None else max_len)
        cache = self.init_cache(z)
        cur = self.special_frame(self.sos_token, B, device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        pad = self.special_frame(self.pad_token, B, device)
        out = []
        for _ in range(max_len):
            if cache['pos'] >= self.max_seq_len or finished.all():
                break
            logits = self.decode_step(cur, cache)
            nxt = self.pick_frame(logits, cur, sample=sample, temperature=temperature, constrained=constrained)
            if pad_after_eos:
                nxt = torch.where(finished.unsqueeze(1), pad, nxt)
            finished = finished | (nxt[:, 0] == self.eos_token)
            out.append(nxt)
            cur = nxt
        if not out:
            return torch.zeros((B, 0, N_FRAME_FIELDS), dtype=torch.long, device=device)
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def sample(self, z, max_len, temperature=1.0, sample=True, device=None, constrained=True):
        device = device or next(self.parameters()).device
        z = z.to(device)
        return self.generate(z, max_len=max_len, temperature=temperature, sample=sample, constrained=constrained)


# ------------------------- VAE wrapper -------------------------
class TransformerVAE(nn.Module):
    def __init__(self, _hparams: SimpleNamespace):
        super().__init__()
        self.encoder = Encoder(_hparams)
        self.decoder = Decoder(_hparams)
        self.latent_dim = _hparams.latent_dim
        self.pad_token = int(_hparams.pad_token)
        self.eos_token = int(_hparams.eos_token)

    def forward(self, frames):
        """frames: [B, T, 4] -> logits [B, T, sum(field_vocab_sizes)], mu, logvar."""
        B = frames.size(0)
        sos = torch.full((B, 1, N_FRAME_FIELDS), self.decoder.sos_token, dtype=torch.long, device=frames.device)
        decoder_input = torch.cat([sos, frames[:, :-1]], dim=1)
        # EOS frames in the decoder input become PAD frames
        is_eos = (decoder_input[..., 0] == self.eos_token).unsqueeze(-1)
        decoder_input = decoder_input.masked_fill(is_eos, self.pad_token)

        mu, logvar = self.encoder(frames)
        z = reparameterize(mu, logvar)
        logits = self.decoder(decoder_input, z)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, z=None, max_len=128, temperature=1.0, sample=True, device=None):
        device = device or next(self.parameters()).device
        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)
        return self.decoder.sample(z.to(device), max_len=max_len, temperature=temperature, sample=sample, device=device)


class EncoderWrapper(nn.Module):
    def __init__(self, encoder, hparams, return_numpy=True):
        super().__init__()
        self.encoder = encoder
        self.hparams = hparams
        self.return_numpy = return_numpy

    def forward(self, frames):
        """frames: [B, T, 4] compound frames -> mu."""
        if not torch.is_tensor(frames):
            frames = torch.tensor(frames, dtype=torch.long, device=next(self.encoder.parameters()).device)
        else:
            frames = frames.to(device=next(self.encoder.parameters()).device, dtype=torch.long)
        mu, _ = self.encoder(frames)
        if self.return_numpy:
            mu = mu.detach().cpu().numpy()
        return mu




# ------------------------- Loss & Training helpers -------------------------
def vae_loss(logits, targets, mu, logvar, beta=1.0, free_bits=0.0, field_slices=None, field_weights=None,
             pad_token=0, eos_token=2):
    """Compound loss.

    logits: [B, T, sum(field_vocab_sizes)], targets: [B, T, 4].
    recon = sum over fields of w_f * CE_f, averaged over the non-PAD frames. On EOS frames only the step head is
    trained (the other fields are ignored). KL is normalised per frame of each sequence.
    Returns (total, recon, kl, perplexity, n_frames, field_ce[4]); field_ce is the mean CE per field over its
    valid targets (detached) and perplexity = exp(mean(field_ce)).
    """
    n_fields = targets.size(-1)
    if field_slices is None:
        raise ValueError('field_slices are required (Decoder.field_slices)')
    if field_weights is None:
        field_weights = [1.0] * n_fields

    step_targets = targets[..., 0]
    frame_mask = step_targets != pad_token  # [B, T]
    n_frames = frame_mask.sum().clamp(min=1)
    n_frames_f = n_frames.to(logits.device, dtype=torch.float32)
    is_eos = step_targets == eos_token

    recon = logits.new_zeros((), dtype=torch.float32)
    field_ce = []
    for f, (start, end) in enumerate(field_slices):
        target_f = targets[..., f]
        if f > 0:
            target_f = target_f.masked_fill(is_eos, pad_token)
        logits_f = logits[..., start:end].reshape(-1, end - start)
        ce_sum = F.cross_entropy(logits_f.float(), target_f.reshape(-1), ignore_index=pad_token, reduction='sum')
        count = (target_f != pad_token).sum().clamp(min=1).to(torch.float32)
        recon = recon + float(field_weights[f]) * ce_sum / n_frames_f
        field_ce.append((ce_sum / count).detach())
    field_ce = torch.stack(field_ce)

    # Dimension-wise KL with per-dimension free bits
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, latent_dim]
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)  # free bits per dimension
    kl_per_sample = kl_per_dim.float().sum(dim=1)  # [B]

    # normalize KL per frame of every sequence
    frames_per_sample = frame_mask.sum(dim=1).clamp(min=1).to(torch.float32)  # [B]
    kl = (kl_per_sample / frames_per_sample).mean()
    kl = beta * kl
    total_loss = recon + kl

    # Kept as a tensor so the training loop does not need a host sync per batch.
    perplexity = torch.exp(torch.clamp(field_ce.mean(), max=20))

    return total_loss, recon, kl, perplexity, n_frames.float(), field_ce


def _atomic_torch_save(obj, path):
    """Write a checkpoint to a temp file first and rename it, so an abort never leaves a half-written file."""
    path = str(path)
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def train_vae(
        model: TransformerVAE, dataloader: DataLoader, _hparams: SimpleNamespace,
    optimizer, device=None, annealing_style='sigmoid', scheduler=None, val_loader=None, resume_state=None,
    on_event=None, should_stop=None, log_interval=25, torch_compile=True):
    """Train the VAE.

    dataloader / val_loader: any iterable of [B, T] token tensors with __len__ (DataLoader or TensorBatcher).
    on_event(dict): optional callback receiving progress events (batch, epoch, checkpoint, log).
    should_stop() -> bool: optional, polled once per batch. When it returns True the current state is saved
        as model_last_*.pt and model_*.pt and the function returns a result dict with stopped=True.
    Returns a dict with timestamp, checkpoint paths, log file and the stopped flag.
    """

    def _emit(event):
        if on_event is not None:
            on_event(event)

    def _stop_requested():
        return bool(should_stop()) if should_stop is not None else False

    # Device selection with priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS device (Apple Silicon)')
    else:
        device = torch.device('cpu')
        print('Using CPU device')



    def _prepare_annealing_points(annealing_list):
        default_points = [(0.0, 0.0), (1.0, 1.0)]
        if not isinstance(annealing_list, (list, tuple)) or len(annealing_list) == 0:
            return default_points

        points = []
        for pair in annealing_list:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            x, y = pair
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError):
                continue
            points.append((min(max(x, 0.0), 1.0), y))

        if len(points) == 0:
            return default_points

        points.sort(key=lambda p: p[0])
        deduped = []
        for x, y in points:
            if deduped and math.isclose(x, deduped[-1][0], rel_tol=0.0, abs_tol=1e-12):
                deduped[-1] = (x, y)
            else:
                deduped.append((x, y))

        if deduped[0][0] > 0.0:
            deduped.insert(0, (0.0, deduped[0][1]))
        if deduped[-1][0] < 1.0:
            deduped.append((1.0, deduped[-1][1]))
        return deduped

    def _bezier_ease(t, slope=0.5):
        # Cubic Hermite / Bezier ease.
        # slope=0 → smoothstep (slow at endpoints, fast in middle)
        # slope=1 → linear
        # Values between give a progressively smoother S-curve as slope approaches 0.
        t = min(max(float(t), 0.0), 1.0)
        return slope * (2*t**3 - 3*t**2 + t) + (-2*t**3 + 3*t**2)

    def _annealed_beta_at_progress(progress, points, slope=0.5):
        p = min(max(float(progress), 0.0), 1.0)
        for idx in range(len(points) - 1):
            x0, y0 = points[idx]
            x1, y1 = points[idx + 1]
            if p <= x1 or idx == len(points) - 2:
                span = max(x1 - x0, 1e-12)
                t = (p - x0) / span
                eased_t = _bezier_ease(t, slope)
                if y0 > 0.0 and y1 > 0.0:
                    # Log-space interpolation with Bezier easing.
                    return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * eased_t)
                return y0 + (y1 - y0) * eased_t
        return points[-1][1]

    epochs = _hparams.n_epochs
    total_train_steps = max(len(dataloader) * epochs, 1)
    annealing_points = _prepare_annealing_points(getattr(_hparams, 'annealing_list', None))
    annealing_slope = float(getattr(_hparams, 'annealing_slope', 0.1))
    # beta_multiplier = _hparams.beta_multiplier
    free_bits = _hparams.free_bits
    loss_history = []

    # Initialize CSV logging and generate model name with timestamp
    timestamp = datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Training timestamp: {timestamp}")
    branch_name = safe_branch_name(get_current_branch())
    log_dir = branch_dir('docs')
    log_file = os.path.join(str(log_dir), f'training_log_{timestamp}.csv')

    # Generate model name with timestamp
    model_dir = branch_dir('models')
    model_name = os.path.join(str(model_dir), f'model_{timestamp}.pt')
    print(f"Using branch-specific output folders for branch '{branch_name}'")

    # Configure performance optimizations based on device
    use_amp = False  # Automatic Mixed Precision
    use_compile = False  # torch.compile for optimization
    scaler = None
    autocast_ctx = None
    # Keep a handle on the uncompiled module: checkpoints must not carry the '_orig_mod.' key prefix.
    raw_model = model

    if device.type == 'cuda':
        # Check for H100/A100 (compute capability 8.0+) for TF32 and other optimizations
        cuda_capability = torch.cuda.get_device_capability(device)
        compute_capability = cuda_capability[0] + cuda_capability[1] / 10

        print(f"CUDA Compute Capability: {compute_capability}")

        # Enable TF32 for H100/A100 (faster matmul operations)
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for faster training on H100/A100")

        # Enable cuDNN benchmarking for optimal kernels
        torch.backends.cudnn.benchmark = True

        # Use Automatic Mixed Precision. bf16 needs no GradScaler; fall back to fp16 + scaler otherwise.
        use_amp = True
        if torch.cuda.is_bf16_supported():
            autocast_ctx = lambda: torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            print("Automatic Mixed Precision (bf16, no grad scaler) enabled")
        else:
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
                scaler = torch.amp.GradScaler('cuda')
                autocast_ctx = lambda: torch.amp.autocast(device_type='cuda')
            else:
                scaler = torch.cuda.amp.GradScaler()
                autocast_ctx = lambda: torch.cuda.amp.autocast()
            print("Automatic Mixed Precision (fp16 + grad scaler) enabled")

        if resume_state is not None and scaler is not None:
            scaler_state = resume_state.get('scaler_state_dict', None)
            if scaler_state is not None:
                try:
                    scaler.load_state_dict(scaler_state)
                    print("Loaded AMP scaler state from checkpoint")
                except Exception as e:
                    print(f"Warning: could not load AMP scaler state: {e}")

        # torch.compile for PyTorch 2.0+ (significant speedup on modern GPUs)
        if not torch_compile:
            print("torch.compile disabled by option")
        elif hasattr(torch, 'compile') and compute_capability >= 8.0:
            compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
            if compiler is None:
                print("torch.compile disabled: no system C compiler found (cc/gcc/clang)")
            else:
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    use_compile = True
                    # If a compiled backend fails at runtime, fall back to eager instead of crashing.
                    from torch import _dynamo
                    _dynamo.config.suppress_errors = True
                    print("torch.compile enabled for additional optimization")
                except Exception as e:
                    print(f"torch.compile not available: {e}")

    elif device.type == 'mps':
        # MPS-specific optimizations for Apple Silicon
        print("MPS optimizations enabled for Apple Silicon")
        # Note: AMP not well supported on MPS yet, keeping FP32

    else:
        print("CPU mode - no hardware-specific optimizations")

    # Compound frames: one cross entropy per field (step, pitch, velocity, deviation).
    field_names = list(getattr(_hparams, 'field_names', None) or ['step', 'pitch', 'velocity', 'deviation'])
    field_slices = raw_model.decoder.field_slices
    field_weights = getattr(_hparams, 'field_loss_weights', None)
    pad_token = int(_hparams.pad_token)
    eos_token = int(_hparams.eos_token)

    def _loss(logits, frames, mu, logvar, beta):
        return vae_loss(logits, frames, mu, logvar, beta=beta, free_bits=free_bits, field_slices=field_slices,
                        field_weights=field_weights, pad_token=pad_token, eos_token=eos_token)

    def _fields_dict(values):
        values = [float(v) for v in values]
        return {name: values[i] for i, name in enumerate(field_names)}

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch_size', 'total_loss', 'reconstruction_loss', 'kl_loss',
            'perplexity', 'avg_frames_per_sequence', 'learning_rate', 'beta',
            'annealing_factor', 'val_total_loss', 'val_reconstruction_loss', 'val_kl_loss',
            'val_perplexity', 'epoch_time_seconds', 'elapsed_time_seconds',
            'remaining_time_seconds', 'saved_cache'
        ] + [f'ce_{name}' for name in field_names] + [f'val_ce_{name}' for name in field_names])

    def _log(text):
        print(text, flush=True)
        _emit({'type': 'log', 'text': str(text)})

    prev_loss = None
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_checkpoint = None
    model.to(device)
    # transfer optimizer state tensors to device:
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    global_step = int(getattr(_hparams, 'global_step', 0) or 0)
    train_step = 0

    best_val_path = os.path.join(model_dir, f'model_best_val_{timestamp}.pt')
    best_train_path = os.path.join(model_dir, f'model_best_train_{timestamp}.pt')
    last_path = os.path.join(model_dir, f'model_last_{timestamp}.pt')

    def _checkpoint_hparams_dict():
        hparams_dict = dict(vars(_hparams))
        hparams_dict.pop('optimizer_state_dict', None)
        hparams_dict.pop('scheduler_state_dict', None)
        hparams_dict.pop('scaler_state_dict', None)
        return hparams_dict

    def _build_checkpoint(epoch, global_step_value, best_val_loss_value, best_train_loss_value, train_loss_value, val_loss_value):
        return {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict() if use_amp and scaler is not None else None,
            'metrics': {
                'train_loss': float(train_loss_value),
                'val_loss': float(val_loss_value) if val_loss_value is not None else None,
                'best_train_loss': float(best_train_loss_value),
                'best_val_loss': float(best_val_loss_value),
            },
            'hparams': _checkpoint_hparams_dict(),
            'epoch': int(epoch),
            'global_step': int(global_step_value),
        }

    def _save_checkpoint(checkpoint, path, kind):
        _atomic_torch_save(checkpoint, path)
        _emit({'type': 'checkpoint', 'kind': kind, 'path': str(path)})

    n_batches = len(dataloader)
    if n_batches == 0:
        raise ValueError('Training data is empty: the data loader has no batches.')
    try:
        n_train_samples = len(dataloader.dataset)
    except (AttributeError, TypeError):
        n_train_samples = int(getattr(_hparams, 'n_samples', 0) or 0)
    n_train_samples = max(int(n_train_samples), 1)

    stopped = False
    lr = optimizer.param_groups[0]['lr']
    weighted_beta = _annealed_beta_at_progress(0.0, annealing_points, annealing_slope)
    train_loss_epoch = float('inf')
    val_loss_epoch = None
    completed_epochs = 0
    perplexity = torch.tensor(1000.0)

    train_start_time = time.time()
    prev_epoch_time = time.time()
    print()
    _log(f"Starting training for {epochs} epochs ({total_train_steps} total steps)")
    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            'loss': torch.zeros((), device=device),
            'rec': torch.zeros((), device=device),
            'kl': torch.zeros((), device=device),
            'n_tokens': torch.zeros((), device=device),
            'fields': torch.zeros(len(field_slices), device=device),
        }
        batch_iteration = 0
        prev_time = time.time()
        last_event_time = time.time()
        last_event_iteration = 0
        print('-' * 100)
        for batch in dataloader:
            if _stop_requested():
                stopped = True
                break
            normalized_step = train_step / max(total_train_steps - 1, 1)
            weighted_beta = _annealed_beta_at_progress(normalized_step, annealing_points, annealing_slope)
            global_step += 1
            train_step += 1
            time_elapsed = time.time() - prev_time
            prev_time = time.time()
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast_ctx():
                    logits, mu, logvar = model(batch)
                    loss, rec, kl, perplexity, token_count, field_ce = _loss(logits, batch, mu, logvar, weighted_beta)

                if scaler is not None:
                    # fp16: scaled backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # bf16: no scaling needed
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                # Standard FP32 training
                logits, mu, logvar = model(batch)
                loss, rec, kl, perplexity, token_count, field_ce = _loss(logits, batch, mu, logvar, weighted_beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_iteration += 1
            if scheduler is not None:
                scheduler.step()
            running['loss'] += loss.detach()
            running['rec'] += rec.detach()
            running['kl'] += kl.detach()
            running['n_tokens'] += token_count.detach()
            running['fields'] += field_ce.detach()

            if on_event is None:
                print(f'\rEpoch {epoch:03d}/{epochs} | Step {batch_iteration:2d}/{n_batches} | step_time: {time_elapsed:.2f}s', end='')
            elif batch_iteration % max(int(log_interval), 1) == 0 or batch_iteration == n_batches:
                # The only host sync of the batch loop: once per event interval.
                loss_value = float(loss.detach())
                now_event = time.time()
                steps_since = max(batch_iteration - last_event_iteration, 1)
                step_time = (now_event - last_event_time) / steps_since
                last_event_time = now_event
                last_event_iteration = batch_iteration
                _emit({
                    'type': 'batch',
                    'epoch': epoch,
                    'step': batch_iteration,
                    'steps': n_batches,
                    'loss': loss_value,
                    'beta': float(weighted_beta),
                    'lr': float(optimizer.param_groups[0]['lr']),
                    'step_time': step_time,
                    'fields': _fields_dict(field_ce.detach().cpu().tolist()),
                })
        print()

        if stopped:
            break

        # Validation evaluation
        val_metrics = None
        if val_loader is not None:
            model.eval()
            val_sums = {
                'loss': torch.zeros((), device=device),
                'rec': torch.zeros((), device=device),
                'kl': torch.zeros((), device=device),
                'perplexity': torch.zeros((), device=device),
                'fields': torch.zeros(len(field_slices), device=device),
            }
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device, non_blocking=True)

                    # Use AMP for validation if enabled
                    if use_amp:
                        with autocast_ctx():
                            logits, mu, logvar = model(val_batch)
                            val_loss, val_rec, val_kl, val_perplexity, _, val_field_ce = _loss(
                                logits, val_batch, mu, logvar, weighted_beta)
                    else:
                        logits, mu, logvar = model(val_batch)
                        val_loss, val_rec, val_kl, val_perplexity, _, val_field_ce = _loss(
                            logits, val_batch, mu, logvar, weighted_beta)

                    val_sums['loss'] += val_loss.detach()
                    val_sums['rec'] += val_rec.detach()
                    val_sums['kl'] += val_kl.detach()
                    val_sums['perplexity'] += val_perplexity.detach()
                    val_sums['fields'] += val_field_ce.detach()

            n_val_batches = max(len(val_loader), 1)
            val_fields = (val_sums.pop('fields') / n_val_batches).cpu().tolist()
            val_metrics = {k: float(v) / n_val_batches for k, v in val_sums.items()}
            val_metrics['fields'] = _fields_dict(val_fields)

        now = time.time()
        epoch_time = now - prev_epoch_time
        prev_epoch_time = now
        lr = optimizer.param_groups[0]['lr']
        _hparams.global_step = global_step
        _hparams.epoch = epoch
        _hparams.global_epoch = global_step // n_batches
        _hparams.learning_rate = lr
        _hparams.weighted_beta = weighted_beta

        train_epoch = {
            'loss': float(running['loss']) / n_batches,
            'rec': float(running['rec']) / n_batches,
            'kl': float(running['kl']) / n_batches,
            'ppl': float(perplexity),
            'n_tokens': float(running['n_tokens']),
            'fields': _fields_dict((running['fields'] / n_batches).cpu().tolist()),
        }
        loss_history.append(train_epoch['loss'])

        train_loss_epoch = train_epoch['loss']
        val_loss_epoch = val_metrics['loss'] if val_metrics is not None else None
        completed_epochs = epoch
        save_cache = False

        # Always save the latest full resume checkpoint.
        last_checkpoint = _build_checkpoint(
            epoch=epoch,
            global_step_value=global_step,
            best_val_loss_value=best_val_loss,
            best_train_loss_value=best_train_loss,
            train_loss_value=train_loss_epoch,
            val_loss_value=val_loss_epoch,
        )
        _save_checkpoint(last_checkpoint, last_path, 'last')

        # Track and save best train checkpoint.
        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch
            best_train_checkpoint = _build_checkpoint(
                epoch=epoch,
                global_step_value=global_step,
                best_val_loss_value=best_val_loss,
                best_train_loss_value=best_train_loss,
                train_loss_value=train_loss_epoch,
                val_loss_value=val_loss_epoch,
            )
            _save_checkpoint(best_train_checkpoint, best_train_path, 'best_train')
            save_cache = True

        # Track and save best validation checkpoint when validation is available.
        if val_loader is not None and val_loss_epoch is not None and val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_val_checkpoint = _build_checkpoint(
                epoch=epoch,
                global_step_value=global_step,
                best_val_loss_value=best_val_loss,
                best_train_loss_value=best_train_loss,
                train_loss_value=train_loss_epoch,
                val_loss_value=val_loss_epoch,
            )
            _save_checkpoint(best_val_checkpoint, best_val_path, 'best_val')
            save_cache = True

        # If no validation loader exists, best_val follows best_train for compatibility.
        if val_loader is None and train_loss_epoch <= best_train_loss:
            best_val_checkpoint = _build_checkpoint(
                epoch=epoch,
                global_step_value=global_step,
                best_val_loss_value=best_train_loss,
                best_train_loss_value=best_train_loss,
                train_loss_value=train_loss_epoch,
                val_loss_value=val_loss_epoch,
            )

        prev_loss = min(train_loss_epoch, prev_loss) if prev_loss is not None else train_loss_epoch

        _log(f"Epoch {epoch}/{epochs} | Train Loss {train_epoch['loss']:.4f} | rec {train_epoch['rec']:.4f} | kl {train_epoch['kl']:.4f} | Epoch_time: {epoch_time:.2f}s | saved cache: {save_cache} | lr: {lr:.10f}")
        if val_metrics is not None:
            _log(f"Val Loss {val_metrics['loss']:.4f} | rec {val_metrics['rec']:.4f} | kl {val_metrics['kl']:.4f} | perplexity {val_metrics['perplexity']:.4f}")
        avg_frames = train_epoch['n_tokens'] / n_train_samples
        fields_text = ' | '.join(f"{name} {value:.4f}" for name, value in train_epoch['fields'].items())
        _log(f"Field CE: {fields_text}")
        print(f"Model Perplexity: {train_epoch['ppl']} | Avg frames per sequence: {avg_frames:.2f}")

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

        _emit({
            'type': 'epoch',
            'epoch': epoch,
            'epochs': epochs,
            'train': {k: train_epoch[k] for k in ('loss', 'rec', 'kl', 'ppl', 'fields')},
            'avg_frames_per_sequence': avg_frames,
            'val': dict(val_metrics, ppl=val_metrics['perplexity']) if val_metrics is not None else None,
            'beta': float(weighted_beta),
            'lr': float(lr),
            'epoch_time': epoch_time,
            'elapsed': elapsed_time,
            'remaining': remaining_time,
            'global_step': global_step,
        })

        # Log metrics to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                _hparams.batch_size,
                f"{train_epoch['loss']:.6f}",
                f"{train_epoch['rec']:.6f}",
                f"{train_epoch['kl']:.6f}",
                f"{train_epoch['ppl']:.4f}",
                f"{avg_frames:.2f}",
                f"{lr:.10f}",
                f"{weighted_beta:.6f}",
                f"{weighted_beta:.6f}",
                f"{val_metrics['loss']:.6f}" if val_metrics is not None else "",
                f"{val_metrics['rec']:.6f}" if val_metrics is not None else "",
                f"{val_metrics['kl']:.6f}" if val_metrics is not None else "",
                f"{val_metrics['perplexity']:.4f}" if val_metrics is not None else "",
                f"{epoch_time:.2f}",
                f"{elapsed_time:.2f}",
                f"{remaining_time:.2f}",
                save_cache
            ] + [f"{v:.6f}" for v in train_epoch['fields'].values()]
              + ([f"{v:.6f}" for v in val_metrics['fields'].values()] if val_metrics is not None else [''] * len(field_names)))

    if stopped:
        # Persist the exact stop state (mid-epoch) as resume checkpoint.
        _hparams.global_step = global_step
        _hparams.learning_rate = optimizer.param_groups[0]['lr']
        _hparams.weighted_beta = weighted_beta
        partial_loss = float(running['loss']) / batch_iteration if batch_iteration > 0 else train_loss_epoch
        stop_checkpoint = _build_checkpoint(
            epoch=completed_epochs,
            global_step_value=global_step,
            best_val_loss_value=best_val_loss,
            best_train_loss_value=best_train_loss,
            train_loss_value=partial_loss,
            val_loss_value=val_loss_epoch,
        )
        _save_checkpoint(stop_checkpoint, last_path, 'last')
        _log(f"Training stopped by request after {completed_epochs} completed epoch(s), {batch_iteration} batch(es) of the current one.")
        final_checkpoint = best_val_checkpoint if best_val_checkpoint is not None else stop_checkpoint
    else:
        final_checkpoint = best_val_checkpoint

    # Save the primary best model at the end of training.
    if final_checkpoint is not None:
        _save_checkpoint(final_checkpoint, model_name, 'final')
        if val_loader is not None:
            print(f"\nBest validation model saved to {model_name} (val loss: {best_val_loss:.4f})")
        else:
            print(f"\nBest training model saved to {model_name} (train loss: {best_train_loss:.4f})")
    else:
        # Fallback: save current model if no best was found
        fallback_checkpoint = _build_checkpoint(
            epoch=completed_epochs,
            global_step_value=global_step,
            best_val_loss_value=best_val_loss,
            best_train_loss_value=best_train_loss,
            train_loss_value=train_loss_epoch,
            val_loss_value=val_loss_epoch,
        )
        _save_checkpoint(fallback_checkpoint, model_name, 'final')
        print(f"\nModel saved to {model_name}")

    print(f"Best-train checkpoint: {best_train_path}")
    print(f"Best-val checkpoint:   {best_val_path}")
    print(f"Last checkpoint:       {last_path}")

    print(f"Training took {(time.time() - train_start_time)/60:.2f} minutes")
    print(f'Training log saved to {log_file}')
    return {
        'timestamp': timestamp,
        'stopped': bool(stopped),
        'epochs_completed': int(completed_epochs),
        'global_step': int(global_step),
        'model_path': model_name,
        'last_path': last_path,
        'best_train_path': best_train_path,
        'best_val_path': best_val_path,
        'log_file': log_file,
    }

def save_model(_hparams, model, model_name='../models/model.pt', scheduler=None, optimizer=None, scaler=None, loss_history=None):
    if loss_history is not None:
        _hparams.loss_history = loss_history

    hparams_dict = dict(vars(_hparams))
    hparams_dict.pop('optimizer_state_dict', None)
    hparams_dict.pop('scheduler_state_dict', None)
    hparams_dict.pop('scaler_state_dict', None)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'hparams': hparams_dict  # Convert SimpleNamespace to dict
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
