import numpy as np
import torch
import torch.nn.functional as F
from modules.utilities import load_model


DATASET_PATH = 'data/global-grooves/dataset-groove.npz'
VALIDATION_SAMPLES = 20
model, hparams = load_model(0, checkpoint_kind='best_val', branch='global-grooves')


def _timing_center_token(_hparams) -> int:
    if hasattr(_hparams, 'center_timing_token'):
        return int(_hparams.center_timing_token)
    if hasattr(_hparams, 'min_timing_token') and hasattr(_hparams, 'n_timing'):
        return int(_hparams.min_timing_token) + int(_hparams.n_timing) // 2
    span = int(_hparams.max_timing_token) - int(_hparams.min_timing_token)
    return int(_hparams.min_timing_token) + span // 2


def _align_generated_to_target(generated: torch.Tensor, target_len: int, pad_token: int) -> torch.Tensor:
    if generated.size(1) > target_len:
        return generated[:, :target_len]
    if generated.size(1) < target_len:
        pad = torch.full(
            (generated.size(0), target_len - generated.size(1)),
            int(pad_token),
            dtype=generated.dtype,
            device=generated.device,
        )
        return torch.cat([generated, pad], dim=1)
    return generated


def _compute_reconstruction_metrics(target_tokens: torch.Tensor, generated_tokens: torch.Tensor, logits: torch.Tensor) -> dict:
    pad_token = int(hparams.pad_token)
    target = target_tokens.to(logits.device)
    generated = _align_generated_to_target(generated_tokens.to(logits.device), target.size(1), pad_token)

    valid_mask = target != pad_token
    n_valid = int(valid_mask.sum().item())

    exact_match = ((generated == target) | (~valid_mask)).all(dim=1)
    exact_match_percent = 100.0 * float(exact_match.float().mean().item())

    token_correct = int(((generated == target) & valid_mask).sum().item())
    token_accuracy = float(token_correct / max(1, n_valid))

    ce_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_token,
        reduction='sum',
    )
    reconstruction_ce = float(ce_sum.item() / max(1, n_valid))

    pitch_mask = (target >= int(hparams.min_pitch_token)) & (target < int(hparams.max_pitch_token))
    pitch_total = int(pitch_mask.sum().item())
    pitch_correct = int(((generated == target) & pitch_mask).sum().item())
    pitch_accuracy = float(pitch_correct / max(1, pitch_total))

    vel_mask = (target >= int(hparams.min_vel_token)) & (target < int(hparams.max_vel_token))
    vel_total = int(vel_mask.sum().item())
    velocity_mae = float(torch.abs(generated[vel_mask] - target[vel_mask]).float().mean().item()) if vel_total > 0 else 0.0

    timing_mask = (target >= int(hparams.min_timing_token)) & (target < int(hparams.max_timing_token))
    timing_total = int(timing_mask.sum().item())
    center = _timing_center_token(hparams)
    if timing_total > 0:
        pred_timing = generated[timing_mask].to(torch.float32) - float(center)
        target_timing = target[timing_mask].to(torch.float32) - float(center)
        micro_timing_mae = float(torch.abs(pred_timing - target_timing).mean().item())
    else:
        micro_timing_mae = 0.0

    return {
        'exact_match_percent': exact_match_percent,
        'token_accuracy': token_accuracy,
        'reconstruction_ce': reconstruction_ce,
        'pitch_accuracy': pitch_accuracy,
        'velocity_mae': velocity_mae,
        'micro_timing_mae': micro_timing_mae,
    }


@torch.no_grad()
def _evaluate_split(split_data: np.ndarray, n_samples: int) -> tuple[dict, int]:
    take = min(n_samples, len(split_data))
    eval_data = split_data[:take]
    data_tensors = torch.tensor(eval_data, dtype=torch.long)

    logits, mu, _logvar = model(data_tensors)
    preds = model.sample(z=mu, max_len=hparams.max_seq_len, temperature=0.1, sample=False)

    return _compute_reconstruction_metrics(data_tensors, preds, logits), take


def _print_comparison_table(train_metrics: dict, val_metrics: dict, train_take: int, val_take: int) -> None:
    print('---' * 10)
    print('Reconstruction Quality Comparison (Original vs Generated)')
    print(f'Samples used -> Train: {train_take}, Validation: {val_take}')
    print('Metric                          | Train         | Validation')
    print('Exact match (% identical seqs) | ' f"{train_metrics['exact_match_percent']:>12.3f}% | {val_metrics['exact_match_percent']:>10.3f}%")
    print('Token accuracy                 | ' f"{train_metrics['token_accuracy']:>12.6f} | {val_metrics['token_accuracy']:>10.6f}")
    print('Cross entropy / recon loss     | ' f"{train_metrics['reconstruction_ce']:>12.6f} | {val_metrics['reconstruction_ce']:>10.6f}")
    print('Pitch accuracy                 | ' f"{train_metrics['pitch_accuracy']:>12.6f} | {val_metrics['pitch_accuracy']:>10.6f}")
    print('Velocity MAE                   | ' f"{train_metrics['velocity_mae']:>12.6f} | {val_metrics['velocity_mae']:>10.6f}")
    print('Micro Timing MAE               | ' f"{train_metrics['micro_timing_mae']:>12.6f} | {val_metrics['micro_timing_mae']:>10.6f}")
    print('---' * 10)


@torch.no_grad()
def main() -> None:
    print(f'Loading dataset: {DATASET_PATH}')
    data = np.load(DATASET_PATH)['data']

    validation_split = float(getattr(hparams, 'validation_split', 0.1))
    val_size = int(len(data) * validation_split)
    train_size = len(data) - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f'Train samples: {len(train_data)}')
    print(f'Validation samples total: {len(val_data)}')

    if len(val_data) == 0:
        raise ValueError('Validation split is empty. Increase dataset size or validation_split.')

    model.eval()
    train_metrics, train_take = _evaluate_split(train_data, VALIDATION_SAMPLES)
    val_metrics, val_take = _evaluate_split(val_data, VALIDATION_SAMPLES)
    _print_comparison_table(train_metrics, val_metrics, train_take, val_take)


if __name__ == '__main__':
    main()
