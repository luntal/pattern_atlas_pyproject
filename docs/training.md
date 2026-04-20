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
    'hparams':              dict,  # all hyperparameters (no optimizer/scaler state)
}
```

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

In addition to checkpoint files, each training run writes a CSV log to `docs/<branch>/training_log_<timestamp>.csv` containing per-epoch metrics: total loss, reconstruction loss, KL loss, perplexity, learning rate, beta, annealing factor, validation metrics, and timing information.

View logs with:

```bash
python src/training_log.py       # latest log
python src/training_log.py 1     # previous log
```
