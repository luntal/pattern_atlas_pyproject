# Model Checkpointing

## Overview

Every call to `train_vae()` produces a set of checkpoint files in the branch-specific model directory (`models/<branch>/`). All checkpoint filenames include the training run timestamp so that multiple training runs coexist without overwriting each other.

---

## Files Written Per Training Run

### 1. `model_best_val_<timestamp>.pt` — Best Validation Checkpoint

Updated whenever the validation total loss reaches a new minimum. If no validation loader is provided this file is not written.

**When it is saved:** once per epoch, only when `val_total_loss < best_val_loss_so_far`.

**Recommended use:** the primary model for evaluation, inference, and deployment. This is the checkpoint that generalizes best to unseen data.

---

### 2. `model_best_train_<timestamp>.pt` — Best Training Checkpoint

Updated whenever the training total loss reaches a new minimum. In practice this is updated during most of the early epochs and then stabilizes.

**When it is saved:** once per epoch, only when `train_total_loss < best_train_loss_so_far`.

**Recommended use:** diagnostics and overfitting analysis. Compare its reconstruction quality against `best_val` to quantify the train/val gap.

---

### 3. `model_last_<timestamp>.pt` — Last Epoch Checkpoint

Overwritten unconditionally at the end of every epoch. Always reflects the model state after the most recently completed epoch, regardless of whether any loss improved.

**When it is saved:** every epoch, always.

**Recommended use:** crash recovery and training resumption. When retraining (`retrain = True` in `train.py`) this is the file to load to continue from where training left off.

---

### 4. `model_<timestamp>.pt` — Primary Inference Model (end-of-training copy)

Written once at the very end of training. It is a copy of the best validation checkpoint (or best training checkpoint if no validation loader was used). This is the file that `load_model()` selects by default when using index-based loading (`previous=0`, `previous=1`, ...).

**When it is saved:** once, after the training loop finishes.

**Recommended use:** standard inference, ONNX export, and RBF server.

---

## Checkpoint Payload

Every checkpoint (all four types above) contains the same set of keys:

```python
{
    'model_state_dict':     ...,   # nn.Module weights
    'optimizer_state_dict': ...,   # Adam state (moments, step counts)
    'scheduler_state_dict': ...,   # LR scheduler state
    'scaler_state_dict':    ...,   # AMP GradScaler state (None on CPU/MPS)
    'epoch':                int,   # epoch at which this checkpoint was saved
    'global_step':          int,   # total optimizer steps taken
    'metrics': {
        'train_loss':       float, # train total loss at this epoch
        'val_loss':         float, # val total loss at this epoch (None if no val)
        'best_train_loss':  float, # best train loss seen so far at save time
        'best_val_loss':    float, # best val loss seen so far at save time
    },
    'hparams':              dict,  # all hyperparameters (no optimizer/scaler state), incl. token_format
}
```

`hparams['token_format']` is `"compound_v1"` for models trained on compound frames. `load_model()` rejects checkpoints without it (old triplet models) with `LegacyCheckpointError` before the weights are loaded.

`hparams` stores epoch and global_step for reference but never stores optimizer, scheduler, or scaler state — those live at the top level of the checkpoint so they can be restored cleanly on resume.

---

## Loading Checkpoints

Use `load_model()` from `src/modules/utilities.py`.

```python
from modules.utilities import load_model

# Named checkpoint selection (recommended)
model, hparams = load_model(checkpoint_kind='best_val')          # latest best_val run
model, hparams = load_model(checkpoint_kind='best_val', previous=1)  # previous run
model, hparams = load_model(checkpoint_kind='best_train')
model, hparams = load_model(checkpoint_kind='last')

# Index-based selection over timestamped model_<timestamp>.pt files
model, hparams = load_model(previous=0)   # most recent end-of-training model
model, hparams = load_model(previous=1)   # one run before that

Named checkpoints (`best_val`, `best_train`, `last`) are excluded from the index-based `previous=N` numbering so the two selection modes do not interfere.

# Full checkpoint access (for retraining/resume)
model, hparams, checkpoint = load_model(checkpoint_kind='last', return_checkpoint=True)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```


---

## Training Log

In addition to checkpoint files, each training run writes a CSV log to `docs/<branch>/training_log_<timestamp>.csv` containing per-epoch metrics: total loss, reconstruction loss, KL loss, perplexity, `avg_frames_per_sequence`, learning rate, beta, annealing factor, validation metrics, timing information and the cross entropy per field (`ce_step`, `ce_pitch`, `ce_velocity`, `ce_deviation` and the `val_ce_*` counterparts). `training_log.py` plots them in an extra panel.

View logs with:

```bash
python src/training_log.py       # latest log
python src/training_log.py 1     # previous log
```

---

# Data and Loss (compound tokens)

Training data are compound frames `[N, max_seq_len, 4]` = `(step, pitch, velocity, deviation)` per note, sorted by `(step, pitch)`, followed by an EOS frame and PAD frames (see `docs/rbf_server_app.md`, section 15). `load_dataset()` only accepts this format; old 2D triplet datasets raise "legacy triplet dataset, re-extract it in the Dataset tab".

The decoder input is the SOS frame followed by the frames shifted by one (EOS frames in the input become PAD). The decoder has four parallel heads; the loss is

$$\text{recon} = \sum_f w_f \cdot \frac{\sum \text{CE}_f}{\text{non-PAD frames}}, \qquad \text{KL} = \beta \cdot \text{mean}_b\left(\frac{\text{KL}_b}{\text{frames}_b}\right)$$

- PAD frames are ignored; on EOS frames only the step head is trained (it decides the sequence end).
- `field_loss_weights` (default `[1, 1, 1, 1]`, order step, pitch, velocity, deviation) is set in `hparams.py` only.
- Perplexity is `exp(mean(field_ce))`.
- Because reconstruction (four decisions per frame) and KL are both normalised per frame, the ratio stays roughly as before and `annealing_list` / `free_bits` are kept as starting values. Check them after the first run.

---

# Training Setup Parameters

## Learning Rate Scheduler

The scheduler is a custom `LambdaLR` that decays the learning rate smoothly from `scheduler_start_lr` to `scheduler_end_lr` over the entire training run (all epochs × steps per epoch). The decay curve is controlled by `scheduler_exp_curve_factor`.

| Parameter | Default | Description |
|---|---|---|
| `scheduler_start_lr` | `4e-4` | Learning rate at step 0. Should match the initial `learning_rate`. |
| `scheduler_end_lr` | `4e-5` | Learning rate at the final step. |
| `scheduler_exp_curve_factor` | `-8.0` | Exponent that shapes the decay curve. `0` → linear decay. Negative values produce a fast initial drop followed by a slow tail (concave-up). Positive values produce a slow start followed by a sharp drop at the end. |
| `scheduler_warmup_ratio` | `0.03` | Fraction of all steps used for a linear warm-up from ~0 to `scheduler_start_lr` (3 % ≈ 2 of 60 epochs). The decay then runs over the remaining steps. `0` = no warm-up. Only used for new trainings; a resume always uses `0` (the Adam moments are already warm) unless the value is set explicitly. |

**Why use a scheduler?** A high initial learning rate allows rapid progress in early epochs while a low final rate lets the optimizer settle into a precise minimum without overshooting. The exponential curve concentrates most of the decay early, which tends to work well for transformer-based models.

**Curve formula** (progress ∈ [0, 1]):

$$\text{lr}(p) = \text{lr}_\text{start} + (\text{lr}_\text{end} - \text{lr}_\text{start}) \cdot \frac{e^{c \cdot p} - 1}{e^{c} - 1}$$

where $c$ is `scheduler_exp_curve_factor`. When $c = 0$ the formula collapses to linear interpolation.

---

## KL Annealing

KL annealing controls the weight of the KL-divergence term in the VAE loss ($\beta$) as a function of training progress. Without annealing, a strong KL penalty early in training pushes the model toward the prior before the decoder has learned anything useful, causing **posterior collapse** — the encoder outputs near-zero KL and the decoder ignores the latent code entirely.

The annealing schedule is defined as a piecewise curve specified in `annealing_list`. Each entry is a `[normalized_position, beta_value]` pair. Normalized position 0 is the first training step and 1 is the last.

| Parameter | Default | Description |
|---|---|---|
| `annealing_list` | `[[0,0],[0.1,0.1],[0.4,0.6],[1.0,0.6]]` | Piecewise control points for the beta schedule. |
| `annealing_slope` | `0.05` | Controls the sharpness of transitions between control points. `0` → linear. Higher values produce a more step-like (sigmoid-shaped) transition. |
| `annealing_steps_norm` | `1` | Scales the total number of annealing steps. `1` = full training duration, `0.5` = first half only. |
| `weighted_beta` | `0.0` | Tracks the current annealed beta during training. Written to checkpoints so retraining can resume the schedule correctly. |

**Interpolation between control points** uses a Bezier-eased cubic Hermite curve in the beta dimension:
- If both adjacent beta values are positive, interpolation is done in log-space for a perceptually uniform curve.
- Otherwise, linear interpolation in beta-space is used.

**Typical schedule interpretation** (default values):
- Steps 0–10 %: beta ramps from 0 → 0.1 (decoder warms up freely).
- Steps 10–40 %: beta ramps from 0.1 → 0.6 (KL penalty gradually applied).
- Steps 40–100 %: beta held at 0.6 (stable regularisation for the remainder of training).

---

## Free Bits

| Parameter | Default | Description |
|---|---|---|
| `free_bits` | `0.1` | Minimum KL contribution per latent dimension (in nats). |

Each latent dimension is individually clamped from below at `free_bits` before summing to the total KL loss. Dimensions that already contribute more than `free_bits` nats are unaffected.

**Why use free bits?** Even with annealing, certain latent dimensions can collapse to near-zero KL and carry no information. The free-bits constraint prevents the model from "switching off" individual dimensions by ensuring each one must encode at least a small amount of information, leading to a richer and more usable latent space.

---

## Weight Decay

| Parameter | Default | Description |
|---|---|---|
| `weight_decay` | `1e-4` | L2 regularisation coefficient applied by the AdamW optimizer. |

Weight decay penalises large parameter magnitudes by adding $\lambda \|\theta\|^2$ to the effective loss. In AdamW this is decoupled from the gradient-based adaptive update, meaning it applies uniformly to all parameters regardless of their gradient history — unlike the original Adam with L2 regularisation, which under-regularises parameters with large gradient variance.

**Why use weight decay?** It discourages the model from relying on a small number of very large weights, improving generalisation and reducing sensitivity to individual training examples. At `1e-4` the effect is mild and primarily acts as a stability mechanism rather than aggressive regularisation.

---

## Speed, stability and stopping options

| Parameter | Default | Description |
|---|---|---|
| `norm_first` | `True` | Pre-norm transformer layers (LayerNorm before attention / feed-forward, plus a final LayerNorm). More stable at higher learning rates; together with the warm-up it needs little tuning. `False` = post-norm like older checkpoints. Part of the architecture: it is taken from the checkpoint when resuming. |
| `max_seq_len` | `128` | Frames (notes + EOS) per 2-bar window the model handles (size of the positional embeddings). Attention cost grows quadratically with it, so keep it close to the real maximum (the Dataset tab shows `max notes` per window). Part of the architecture. |
| `last_checkpoint_interval` | `3` | `model_last_*.pt` is written every N epochs, and always at the last epoch, on a stop and on early stopping. A crash loses at most N-1 epochs of the resume checkpoint. |
| `early_stopping` | `False` | Stop when the validation loss (training loss without validation split) stops improving. |
| `early_stopping_patience` | `8` | Epochs without improvement before the run is stopped. |
| `early_stopping_min_delta` | `0.001` | Smallest decrease of the monitored loss that counts as an improvement. |

**Fused attention.** Encoder and decoder layers (`FusedEncoderLayer`, `FusedDecoderLayer` in `modules/model.py`) use `F.scaled_dot_product_attention` (flash / memory-efficient kernels) instead of `nn.Transformer*Layer`. Parameter names are identical to the torch layers, so existing checkpoints load unchanged and give the same outputs (checked in `tests/test_training_options.py`). The decoder is purely causal without a padding mask: PAD frames sit behind the last real frame, so no real position attends to one and no loss term changes. That allows `is_causal=True` (flash kernel). The encoder passes a boolean key-padding mask for the PAD frames.

**One-pass loss.** `vae_loss` gathers the logits of the four fields into one padded tensor `[B, T, 4, V_max]` and computes a single `logsumexp` / gather instead of four cross entropies on slices. Values, per-field CE and gradients equal the former per-field loop.

**Pre-norm and old checkpoints.** A checkpoint without `norm_first` is a post-norm model and is rebuilt as such. When resuming, `norm_first` and `max_seq_len` come from the checkpoint (the GUI locks them), and the next checkpoint keeps them.

**Early stopping and beta annealing.** The total loss rises while beta is annealed up, so an improvement cannot be judged then. Monitoring therefore starts at the first epoch end after beta has settled, i.e. after the last point of `annealing_list` that still changes beta (with the default list at 40 % of the run). That epoch is the reference; every later epoch without an improvement of at least `early_stopping_min_delta` counts towards the patience. With a beta curve that rises until the very end, early stopping never fires. When it fires, `last` and `model_<timestamp>.pt` (best validation model) are written and the result has `early_stopped = True`; the stop is logged (`Early stopping after epoch N …`).

**Dataset length vs. `max_seq_len`.** A dataset with another number of frames than the model's `max_seq_len` is padded with PAD frames or cut after time (EOS as last frame) when it is loaded (`compound.resize_frames_array`); the log names the number of cut sequences. In the Dataset tab the length of a new dataset is set with *max_seq_len (frames)* of the extraction (default: `hparams.py`).

Test: `python tests/test_training_options.py`.

---

## Training from the Server App

The settings above can also be set in the **Training** tab of `rbf_server_app` (see `docs/rbf_server_app.md`, section 13). `train.py` and the server share one implementation: `modules/training.run_training(config, on_event, should_stop)`. `train.py` is only a thin CLI wrapper that builds a config from `hparams.py` (`retrain` switch and overrides are still at the top of the script); the server starts `train_worker.py`, which calls the same function and reports JSON events.

### Changes to the training loop

- **Tensor batching:** the dataset (small int tensors) is moved to the device once and cut into batches by an index permutation (`TensorBatcher`). The DataLoader and its worker processes are gone, so there is no pickling, no per-batch host-to-device copy and no worker start-up time.
- `optimizer.zero_grad(set_to_none=True)` and fused AdamW on CUDA.
- No `.item()` sync per batch: metrics stay tensors and are read once per epoch (and once per event interval when a callback is used).
- bf16 autocast on CUDA when supported (no GradScaler needed), otherwise fp16 with scaler as before. `torch.compile` is a switch (`torch_compile`); checkpoints always hold the uncompiled `state_dict`.
- Checkpoints are written atomically (temporary file, then `os.replace`), so a stop never leaves a half-written file.
- Fixes: scheduler works with `validation_split = 0`; `avg_frames_per_sequence` (formerly `avg_tokens_per_sequence`) is divided by the real training-set size instead of `n_samples`; `train_vae` no longer needs the DataLoader's `.dataset`.
- `batch` and `epoch` events carry `fields: {step, pitch, velocity, deviation}` (train and val) for the per field chart of the Training tab.

### Stop and resume

`train_vae(..., should_stop=...)` polls the flag once per batch. On stop it saves the exact state as `model_last_*.pt` (epoch counter, `global_step`, optimizer, scheduler) and writes `model_*.pt`, then returns `stopped = True`. A later *Resume* with `last` continues from there. Tests: `tests/test_training_stop.py`, `tests/test_hparams_io.py`, `tests/test_train_server.py`.



## Multi-GPU training (DistributedDataParallel)

The Training tab lists the CUDA GPUs of the server (name, memory) in the *Options* box. Click a GPU to use it or to
leave it out. Default: only GPU 0; at least one GPU always stays selected. Without CUDA the list says so and the
training runs on MPS / CPU as before.

- **One GPU:** unchanged single-process training.
- **Several GPUs:** `train_worker.py` (rank 0) starts one extra process per further GPU (`RANK`, `WORLD_SIZE`,
  `MASTER_ADDR`, `MASTER_PORT` in the environment, NCCL backend). `modules/distributed.py` holds the helpers.
- **Batch size is global.** `batch_size = 512` on 2 GPUs means 256 per GPU, so the learning-rate and beta schedules
  behave like on one GPU. The batch must be at least the number of GPUs. Per step at most `world_size - 1` samples are
  left out (different ones every epoch) so that all ranks run equally sized steps.
- **Metrics, validation and stop** are synchronised over the ranks. Only rank 0 writes checkpoints (without a `module.`
  prefix, so they load like single-GPU checkpoints), the CSV log and the events. `epoch` events carry `world_size`.
- **Failures:** if a rank dies, rank 0 reports an error and kills the others. If rank 0 is killed, the other ranks
  exit by themselves.
- `torch.compile` uses mode `default` under DDP (CUDA graphs and DDP hooks do not combine well).
- With a small model the gradient synchronisation costs a part of the gain; a larger batch size usually helps more than
  an additional GPU. NVLink needs no code change, NCCL uses it automatically.
- Test without GPUs: `python tests/test_ddp.py` (CPU processes with the gloo backend, option `ddp_cpu_processes`).
- Not tested on real GPUs / NCCL yet.


## Throughput and GPU measurement

`nvidia-smi` showing 100 % only means that some kernel was running; it does not say how well the GPU is filled.
`train_vae` therefore measures every epoch (training phase only, validation and checkpoints excluded):

- **samples/s** and **ms/step** (`batch_size` is the global batch). The loss is read back once at the end of the
  phase, so queued GPU work is included in the time.
- **GPU power, utilisation and memory** of the selected GPUs, sampled once per second with `nvidia-smi`
  (`modules/perf_monitor.py`). Without `nvidia-smi` only the throughput is reported. `RBF_PERF_MONITOR=0` switches the
  sampling off. GPU indices are the physical ones; with `CUDA_VISIBLE_DEVICES` they can differ from the torch indices.
- Output: a log line `Throughput: ... samples/s | GPU0 ... W, util ... %, ... GB`, the `perf` object of the `epoch`
  event (shown in the Training tab status), and the CSV columns `train_samples_per_s`, `train_time_seconds`,
  `gpu_power_w` (sum over GPUs), `gpu_util_percent` (mean).

How to use it: run a few epochs per batch size (e.g. 512 / 1024 / 2048). If samples/s stops growing and the power draw
stays far below the card's limit (H100: ~300-700 W), the step is latency bound and a larger batch or a larger model is
free. If samples/s still grows nearly linearly, the GPU was not yet saturated.
