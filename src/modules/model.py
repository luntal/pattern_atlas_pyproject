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

        self.n_memory_tokens = getattr(_hparams, 'n_memory_tokens', 1)
        self.z_to_mem = nn.Linear(_hparams.latent_dim, self.emb_size * self.n_memory_tokens)
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


        memory = self.z_to_mem(z).view(B, self.n_memory_tokens, self.emb_size)
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
        memory = self.z_to_mem(z).view(B, self.n_memory_tokens, self.emb_size)
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
        self.pad_token = _hparams.pad_token

    def forward(self, tokens):
        B, T = tokens.size()
        sos = torch.full((B, 1), self.decoder.sos_token, dtype=torch.long, device=tokens.device)
        decoder_input = torch.cat([sos, tokens[:, :-1]], dim=1)
        # mask the input eos token
        decoder_input[decoder_input == 2] = 0
        # decoder_input = torch.where(decoder_input == 2, torch.zeros_like(decoder_input), decoder_input)
        beat_tokens_enc = make_beat_tokens(tokens, step_token=int(self.step_token), pad_token=int(self.pad_token))

        beat0 = torch.zeros((B, 1), dtype=torch.long, device=tokens.device)
        beat_tokens_dec = torch.cat([beat0, beat_tokens_enc[:, :-1]], dim=1)

        # print('enc input shape: ', tokens.shape)
        mu, logvar = self.encoder(tokens, beat_tokens_enc)
        z = reparameterize(mu, logvar)
        logits = self.decoder(decoder_input, z, beat_tokens_dec)
        return logits, mu, logvar

    # def make_beat_tokens(self, tokens: torch.Tensor, step_token: int, pad_token: int) -> torch.Tensor:
    #     # tokens: [B, T] (long)
    #     is_step = (tokens == step_token).to(torch.long)  # [B, T] in {0,1}
    #     beat_tokens = torch.cumsum(is_step, dim=1)  # [B, T]
    #     beat_tokens += 1
    #     beat_tokens[:,0] = 0
    #     # padding with zeros
    #     pad_mask = (tokens == pad_token)
    #     beat_tokens = beat_tokens.masked_fill(pad_mask, 0)
    #     return beat_tokens

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

    def forward(self, tokens):
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.long, device=next(self.encoder.parameters()).device)
        else:
            tokens = tokens.to(device=next(self.encoder.parameters()).device, dtype=torch.long)

        beat_tokens = make_beat_tokens(tokens, step_token=int(self.hparams.step_token), pad_token=int(self.hparams.pad_token))
        mu, _ = self.encoder(tokens, beat_tokens)
        if self.return_numpy:
            mu = mu.detach().cpu().numpy()
        return mu



def make_beat_tokens(tokens: torch.Tensor, step_token: int, pad_token: int) -> torch.Tensor:
        # tokens: [B, T] (long)
        is_step = (tokens == step_token).to(torch.long)  # [B, T] in {0,1}
        beat_tokens = torch.cumsum(is_step, dim=1)  # [B, T]
        beat_tokens += 1
        beat_tokens[:,0] = 0
        # padding with zeros
        pad_mask = (tokens == pad_token)
        beat_tokens = beat_tokens.masked_fill(pad_mask, 0)
        return beat_tokens




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
    token_count = mask.sum().clamp(min=1)  # int64, safe from float16 overflow
    token_count_f = token_count.to(logits.device, dtype=logits.dtype)  # cast only for loss division
    ce_sum = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token, reduction="sum")
    # loss per token
    recon = ce_sum / token_count_f

    # Dimension-wise KL with per-dimension free bits
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, latent_dim]
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)  # free bits per dimension
    kl_per_sample = kl_per_dim.sum(dim=1)  # [B]

    # number of valid tokens per sample
    tokens_per_sample = (targets != pad_token).sum(dim=1).clamp(min=1).to(logits.dtype)  # [B]

    # normalize KL per token
    kl_per_token = kl_per_sample / tokens_per_sample  # [B]
    kl = kl_per_token.mean()
    kl = beta * kl
    total_loss = recon + kl

    # info = (
    #     f"CE/token: {recon.item():.4f} "
    #     f"PPL: {torch.exp(torch.clamp(recon, max=20)).item():.4f} "
    # )
    perplexity = torch.exp(torch.clamp(recon, max=20)).item()

    return total_loss, recon, kl, perplexity, token_count.float()

def train_vae(
        model: TransformerVAE, dataloader: DataLoader, _hparams: SimpleNamespace,
        optimizer, device=None, annealing_style='sigmoid', scheduler=None, val_loader=None):
    
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

    def _exp_ease(t, k=6.0):
        t = min(max(float(t), 0.0), 1.0)
        if math.isclose(k, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return t
        return (math.exp(k * t) - 1.0) / (math.exp(k) - 1.0)

    def _annealed_beta_at_progress(progress, points):
        p = min(max(float(progress), 0.0), 1.0)
        for idx in range(len(points) - 1):
            x0, y0 = points[idx]
            x1, y1 = points[idx + 1]
            if p <= x1 or idx == len(points) - 2:
                span = max(x1 - x0, 1e-12)
                t = (p - x0) / span
                if y0 > 0.0 and y1 > 0.0:
                    # Log-space interpolation gives a smooth exponential segment.
                    return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * t)
                eased_t = _exp_ease(t)
                return y0 + (y1 - y0) * eased_t
        return points[-1][1]

    epochs = _hparams.n_epochs
    total_train_steps = max(len(dataloader) * epochs, 1)
    annealing_points = _prepare_annealing_points(getattr(_hparams, 'annealing_list', None))
    # beta_multiplier = _hparams.beta_multiplier
    free_bits = _hparams.free_bits
    loss_history = []

    # Initialize CSV logging and generate model name with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

        # Use Automatic Mixed Precision (FP16) for H100
        # H100 has excellent FP16 and even FP8 support
        use_amp = True
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda')
            autocast_ctx = lambda: torch.amp.autocast(device_type='cuda')
        else:
            scaler = torch.cuda.amp.GradScaler()
            autocast_ctx = lambda: torch.cuda.amp.autocast()
        print("Automatic Mixed Precision (AMP) enabled")

        # torch.compile for PyTorch 2.0+ (significant speedup on modern GPUs)
        if hasattr(torch, 'compile') and compute_capability >= 8.0:
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

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'batch_size', 'total_loss', 'reconstruction_loss', 'kl_loss',
            'perplexity', 'avg_tokens_per_sequence', 'learning_rate', 'beta',
            'annealing_factor', 'val_total_loss', 'val_reconstruction_loss', 'val_kl_loss',
            'val_perplexity', 'epoch_time_seconds', 'elapsed_time_seconds',
            'remaining_time_seconds', 'saved_cache'
        ])

    prev_loss = None
    best_loss = float('inf')
    best_model_state = None
    model.to(device)
    # transfer optimizer state tensors to device:
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)



    global_step = 0 + _hparams.global_step
    train_step = 0

    train_start_time = time.time()
    prev_epoch_time = time.time()
    print()
    print(f"Starting training for {epochs} epochs ({total_train_steps} total steps)")
    for epoch in range(1, epochs + 1):
        model.train()
        running = {'loss': 0.0, 'rec': 0.0, 'kl': 0.0, 'n_tokens': 0.0}
        batch_iteration = 0
        prev_time = time.time()
        perplexity = 1000
        print('-' * 100)
        for batch in dataloader:
            normalized_step = train_step / max(total_train_steps - 1, 1)
            weighted_beta = _annealed_beta_at_progress(normalized_step, annealing_points)
            global_step += 1
            train_step += 1
            # weighted_beta = beta_multiplier * annealed_beta
            annealing_factor = weighted_beta
            time_elapsed = time.time() - prev_time
            prev_time = time.time()
            batch = batch.to(device)
            optimizer.zero_grad()

            # Use automatic mixed precision if enabled
            if use_amp:
                with autocast_ctx():
                    logits, mu, logvar = model(batch)
                    loss, rec, kl, perplexity, token_count = vae_loss(logits, batch, mu, logvar, beta=weighted_beta, free_bits=free_bits, pad_token=_hparams.pad_token)

                # Scaled backward pass for mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 training
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

        # Validation evaluation
        val_metrics = {'loss': 0.0, 'rec': 0.0, 'kl': 0.0, 'perplexity': 0.0}
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)

                    # Use AMP for validation if enabled
                    if use_amp:
                        with autocast_ctx():
                            logits, mu, logvar = model(val_batch)
                            val_loss, val_rec, val_kl, val_perplexity, _ = vae_loss(
                                logits, val_batch, mu, logvar, beta=weighted_beta,
                                free_bits=free_bits, pad_token=_hparams.pad_token
                            )
                    else:
                        logits, mu, logvar = model(val_batch)
                        val_loss, val_rec, val_kl, val_perplexity, _ = vae_loss(
                            logits, val_batch, mu, logvar, beta=weighted_beta,
                            free_bits=free_bits, pad_token=_hparams.pad_token
                        )

                    val_metrics['loss'] += val_loss.detach()
                    val_metrics['rec'] += val_rec.detach()
                    val_metrics['kl'] += val_kl.detach()
                    val_metrics['perplexity'] += val_perplexity

            n_val_batches = len(val_loader)
            val_metrics['loss'] /= n_val_batches
            val_metrics['rec'] /= n_val_batches
            val_metrics['kl'] /= n_val_batches
            val_metrics['perplexity'] /= n_val_batches

        now = time.time()
        epoch_time = now - prev_epoch_time
        prev_epoch_time = now
        lr = optimizer.param_groups[0]['lr']
        _hparams.global_step = global_step
        _hparams.global_epoch = global_step // n_batches
        _hparams.learning_rate = lr
        _hparams.weighted_beta = weighted_beta
        _hparams.optimizer_state_dict = optimizer.state_dict()
        loss_history.append(running['loss']/n_batches)

        # Track best model based on validation loss (or training loss if no validation)
        current_loss = val_metrics['loss'] if val_loader is not None else running['loss']/n_batches
        save_cache = False
        if current_loss < best_loss:
            best_loss = current_loss
            # Save best model state
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'hparams': vars(_hparams),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                # 'loss_history': loss_history
            }
            # Save cache checkpoint immediately
            cache_path = os.path.join(model_dir, 'model_cache.pt')
            torch.save(best_model_state, cache_path)
            save_cache = True

        prev_loss = min(running['loss']/n_batches, prev_loss) if prev_loss is not None else running['loss']/n_batches

        print(f"Train Loss {running['loss']/n_batches:.4f} | rec {running['rec']/n_batches:.4f} | kl {running['kl']/n_batches:.4f} | Epoch_time: {epoch_time:.2f}s | saved cache: {save_cache} | lr: {lr:.10f}")
        if val_loader is not None:
            print(f"Val Loss {val_metrics['loss']:.4f} | rec {val_metrics['rec']:.4f} | kl {val_metrics['kl']:.4f} | perplexity {val_metrics['perplexity']:.4f}")
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
                f"{lr:.10f}",
                f"{weighted_beta:.6f}",
                f"{annealing_factor:.6f}",
                f"{val_metrics['loss'].item():.6f}" if val_loader is not None else "",
                f"{val_metrics['rec'].item():.6f}" if val_loader is not None else "",
                f"{val_metrics['kl'].item():.6f}" if val_loader is not None else "",
                f"{val_metrics['perplexity']:.4f}" if val_loader is not None else "",
                f"{epoch_time:.2f}",
                f"{elapsed_time:.2f}",
                f"{remaining_time:.2f}",
                save_cache
            ])

        

    # Save the best model at the end of training
    if best_model_state is not None:
        torch.save(best_model_state, model_name)
        print(f"\nBest model saved to {model_name} (loss: {best_loss:.4f})")
    else:
        # Fallback: save current model if no best was found
        save_model(_hparams, model, model_name, scheduler, optimizer, loss_history)
        print(f"\nModel saved to {model_name}")

    print(f"Training took {(time.time() - train_start_time)/60:.2f} minutes")
    print(f'Training log saved to {log_file}')

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
