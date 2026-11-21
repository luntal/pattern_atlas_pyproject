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
