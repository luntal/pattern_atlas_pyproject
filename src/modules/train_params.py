"""Whitelist and type coercion for training / model parameters that can be set from the GUI or train.py.

Kept free of heavy imports (no torch) so it can also be used by hparams_io and the server.
"""
import math

TRAIN_KEYS = [
    'dataset_name', 'validation_split', 'batch_size', 'n_epochs', 'learning_rate',
    'scheduler_start_lr', 'scheduler_end_lr', 'scheduler_exp_curve_factor', 'weight_decay',
    'n_samples', 'free_bits', 'annealing_list', 'annealing_slope', 'seed',
]
MODEL_KEYS = ['d_model', 'n_heads', 'n_layers', 'ff_dim', 'dropout', 'latent_dim', 'n_memory_tokens']
ALLOWED_KEYS = TRAIN_KEYS + MODEL_KEYS

# Architecture keys are taken from the checkpoint when resuming.
ARCHITECTURE_KEYS = list(MODEL_KEYS)

PARAM_TYPES = {
    'dataset_name': str,
    'validation_split': float,
    'batch_size': int,
    'n_epochs': int,
    'learning_rate': float,
    'scheduler_start_lr': float,
    'scheduler_end_lr': float,
    'scheduler_exp_curve_factor': float,
    'weight_decay': float,
    'n_samples': int,
    'free_bits': float,
    'annealing_list': list,
    'annealing_slope': float,
    'seed': int,
    'd_model': int,
    'n_heads': int,
    'n_layers': int,
    'ff_dim': int,
    'dropout': float,
    'latent_dim': int,
    'n_memory_tokens': int,
}


def coerce_param(key, value):
    """Convert a raw (e.g. JSON) value to the expected type. Raises ValueError with a readable message."""
    if key not in PARAM_TYPES:
        raise ValueError(f"Parameter '{key}' is not allowed")
    kind = PARAM_TYPES[key]
    if kind is str:
        text = str(value).strip()
        if not text:
            raise ValueError('must not be empty')
        return text
    if kind is list:
        if not isinstance(value, (list, tuple)):
            raise ValueError('must be a list of [position, beta] pairs')
        points = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('each entry must be a [position, beta] pair')
            try:
                x, y = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                raise ValueError('position and beta must be numbers')
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError('position and beta must be finite')
            points.append([x, y])
        return points
    if isinstance(value, bool):
        raise ValueError('must be a number')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('must be a number')
    if not math.isfinite(number):
        raise ValueError('must be finite')
    if kind is int:
        if abs(number - round(number)) > 1e-9:
            raise ValueError('must be an integer')
        return int(round(number))
    return number
