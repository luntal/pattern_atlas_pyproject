"""Shared training core used by train.py (CLI) and train_worker.py (server subprocess).

run_training(config, on_event, should_stop) builds model, data and optimizer from a TrainConfig dict and calls train_vae.
Data are compound frames [N, T, 4] (modules.compound); legacy triplet datasets and checkpoints are rejected
(load_dataset / load_model raise with a clear message).

TrainConfig:
    mode: "new" | "resume"
    resume: {mode: best_val|best_train|last|none, previous: int, name: str|None,
             continue_beta: bool, end_lr_factor: float|None}      (only for mode == "resume")
    overrides: {key: value}   (only keys from modules.train_params.ALLOWED_KEYS)
    options: {torch_compile: bool, log_interval: int}
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch

from modules.model import TransformerVAE, train_vae
from modules.train_params import ALLOWED_KEYS, ARCHITECTURE_KEYS, coerce_param
from modules.utilities import load_dataset, load_model, load_scheduler


def pick_device() -> torch.device:
    """Same priority as train_vae: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TensorBatcher:
    """Batches a frame tensor [N, T, 4] (any tensor batched over axis 0) that already lives on the target device.

    Replaces DataLoader + worker processes for small integer datasets: no pickling, no host-to-device
    copies per batch, no worker start-up time. Behaves like a DataLoader with drop_last=False.
    """

    def __init__(self, data: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 0):
        self.data = data
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed))

    @property
    def dataset(self):
        return self.data

    def __len__(self):
        return (self.data.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.data.shape[0]
        if self.shuffle:
            order = torch.randperm(n, generator=self._generator).to(self.data.device)
        else:
            order = None
        for start in range(0, n, self.batch_size):
            if order is None:
                yield self.data[start:start + self.batch_size]
            else:
                yield self.data[order[start:start + self.batch_size]]


def default_hparams() -> SimpleNamespace:
    from hparams import hparams as base
    return SimpleNamespace(**copy.deepcopy(vars(base)))


def validate_hparams(hp):
    if int(hp.d_model) % int(hp.n_heads) != 0:
        raise ValueError(f"d_model ({hp.d_model}) must be divisible by n_heads ({hp.n_heads})")
    if not (0.0 <= float(hp.validation_split) < 1.0):
        raise ValueError('validation_split must be in [0, 1)')
    if int(hp.batch_size) < 1 or int(hp.n_epochs) < 1:
        raise ValueError('batch_size and n_epochs must be >= 1')


def _apply_overrides(hp, overrides, skip_keys=()):
    skipped = []
    for key, raw in (overrides or {}).items():
        if key not in ALLOWED_KEYS:
            raise ValueError(f"Parameter '{key}' is not allowed")
        if key in skip_keys:
            skipped.append(key)
            continue
        setattr(hp, key, coerce_param(key, raw))
    return skipped


def run_training(config, on_event=None, should_stop=None):
    """Run one training. Returns the result dict of train_vae. Raises on invalid configuration."""
    config = dict(config or {})
    mode = config.get('mode', 'new')
    overrides = dict(config.get('overrides') or {})
    options = dict(config.get('options') or {})
    resume_cfg = dict(config.get('resume') or {})

    def _emit(event):
        if on_event is not None:
            on_event(event)

    def _status(text):
        print(text, flush=True)
        _emit({'type': 'status', 'text': text})

    if mode not in ('new', 'resume'):
        raise ValueError(f"Unknown mode '{mode}'")

    device = pick_device()
    resume_checkpoint = None
    vae = None

    if mode == 'resume':
        _status('Loading checkpoint ...')
        kind = resume_cfg.get('mode', 'last')
        kind = None if kind in (None, 'none') else kind
        vae, hp, resume_checkpoint = load_model(
            previous=int(resume_cfg.get('previous', 0) or 0),
            onnx=False,
            return_checkpoint=True,
            checkpoint_kind=kind,
            name=resume_cfg.get('name') or None,
        )
        hp = SimpleNamespace(**copy.deepcopy(vars(hp)))
        # Keys that the checkpoint does not carry (older checkpoints) fall back to the defaults.
        for key, value in vars(default_hparams()).items():
            if not hasattr(hp, key):
                setattr(hp, key, value)
        skipped = _apply_overrides(hp, overrides, skip_keys=ARCHITECTURE_KEYS)
        if skipped:
            _status(f"Architecture parameters come from the checkpoint, ignoring: {', '.join(skipped)}")
        if resume_cfg.get('continue_beta') and 'annealing_list' not in overrides:
            w_beta = float(getattr(hp, 'weighted_beta', 0.0) or 0.0)
            hp.annealing_list = [[0.0, w_beta], [0.5, w_beta], [1.0, w_beta]]
        if 'scheduler_start_lr' not in overrides and resume_cfg.get('start_lr') is not None:
            hp.scheduler_start_lr = float(resume_cfg['start_lr'])
        if 'scheduler_end_lr' not in overrides and resume_cfg.get('end_lr_factor') is not None:
            hp.scheduler_end_lr = float(hp.learning_rate) * float(resume_cfg['end_lr_factor'])
    else:
        hp = default_hparams()
        _apply_overrides(hp, overrides)
        hp.global_step = 0
        hp.global_epoch = 0

    validate_hparams(hp)
    torch.manual_seed(int(hp.seed))
    np.random.seed(int(hp.seed) % (2 ** 32))

    if vae is None:
        vae = TransformerVAE(hp)
    vae.to(device)

    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    _status(f'Model parameters: {total_params:,} | device: {device}')

    _status(f'Loading dataset {hp.dataset_name} ...')
    data, _metadata = load_dataset(hp)

    val_size = int(len(data) * float(hp.validation_split))
    train_size = len(data) - val_size
    if train_size < 1:
        raise ValueError('No training samples left after the validation split')
    train_tensor = torch.from_numpy(np.ascontiguousarray(data[:train_size])).long().to(device)
    val_tensor = torch.from_numpy(np.ascontiguousarray(data[train_size:])).long().to(device) if val_size > 0 else None
    _status(f'Train samples: {train_size} | validation samples: {val_size}')

    loader = TensorBatcher(train_tensor, hp.batch_size, shuffle=True, seed=int(hp.seed))
    val_loader = TensorBatcher(val_tensor, hp.batch_size, shuffle=False) if val_tensor is not None else None

    optim = torch.optim.AdamW(
        vae.parameters(),
        lr=float(hp.learning_rate),
        weight_decay=float(getattr(hp, 'weight_decay', 0.0)),
        fused=(device.type == 'cuda'),
    )

    if resume_checkpoint is not None:
        optim_state = resume_checkpoint.get('optimizer_state_dict', None)
        if optim_state is not None:
            try:
                optim.load_state_dict(optim_state)
            except Exception as exc:
                _status(f'Warning: optimizer state could not be restored ({exc}), starting with a fresh optimizer state')
        optim.param_groups[0]['lr'] = float(hp.learning_rate)

    scheduler = load_scheduler(optim, loader, hp)
    if resume_checkpoint is not None:
        scheduler_state = resume_checkpoint.get('scheduler_state_dict', None)
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                _status(f'Warning: scheduler state could not be restored ({exc})')

    _status('Training ...')
    return train_vae(
        vae, loader, hp, optim,
        scheduler=scheduler,
        val_loader=val_loader,
        resume_state=resume_checkpoint,
        on_event=on_event,
        should_stop=should_stop,
        log_interval=int(options.get('log_interval', 25) or 25),
        torch_compile=bool(options.get('torch_compile', True)),
    )
