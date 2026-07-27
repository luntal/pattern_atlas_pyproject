import glob
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from modules.model import ONNXWrapper, TransformerVAE, reparameterize

"""
Evaluate reconstruction quality on a sample subset without visualization or MIDI playback.
"""


##########################
# Configuration
##########################
ONNX = False
# model_name = None  # Example: 'model_2026-04-08_09-52-22.pt'
model_name = 'model_2026-04-08_09-52-22.pt'
dataset_name = 'dataset-groove.npz'
sample_skip = 0
sample_take = 100  # Set <= 0 to use the full dataset after skipping.
position_tolerance = 24  # Match token within +/- this many positions.
pitch_tolerance_steps = 1
velocity_tolerance_percent = 10
micro_timing_tolerance_percent = 10


##########################
# Helpers
##########################
def get_safe_branch_name(project_root: str) -> str:
    branch_name = 'unknown-branch'
    try:
        result = subprocess.run(
            ['git', '-C', project_root, 'rev-parse', '--abbrev-ref', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        )
        branch_name = result.stdout.strip() or 'unknown-branch'
        if branch_name == 'HEAD':
            detached_result = subprocess.run(
                ['git', '-C', project_root, 'rev-parse', '--short', 'HEAD'],
                check=True,
                capture_output=True,
                text=True,
            )
            commit_hash = detached_result.stdout.strip() or 'unknown'
            branch_name = f'detached-{commit_hash}'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return branch_name.replace('/', '_').replace(' ', '_')


def resolve_model_path(project_root: str, safe_branch_name: str, model_name_override: str | None) -> str:
    models_dir = os.path.join(project_root, 'models', safe_branch_name)
    model_files = glob.glob(os.path.join(models_dir, '*.pt'))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    selected_model = None
    if model_name_override is not None:
        candidate_name = os.path.basename(model_name_override)
        candidate_path = os.path.join(models_dir, candidate_name)
        if os.path.isfile(candidate_path):
            selected_model = candidate_path
        elif not candidate_name.endswith('.pt') and os.path.isfile(candidate_path + '.pt'):
            selected_model = candidate_path + '.pt'
        else:
            print(
                f"Requested model_name '{model_name_override}' not found in {models_dir}; "
                "falling back to latest model."
            )

    if selected_model is None:
        selected_model = max(model_files, key=os.path.getmtime)

    return selected_model


def resolve_dataset_path(project_root: str, safe_branch_name: str, dataset_filename: str) -> str:
    if not dataset_filename.endswith('.npz'):
        dataset_filename += '.npz'

    dataset_dir = os.path.join(project_root, 'data', safe_branch_name)
    dataset_path = os.path.join(dataset_dir, dataset_filename)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return dataset_path


def windowed_position_accuracy(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    tolerance: int,
    token_matcher=None,
) -> tuple[float, int]:
    total_reference = 0
    total_matches = 0

    batch_size = target_tokens.shape[0]
    for i in range(batch_size):
        pred_seq = pred_tokens[i]
        target_seq = target_tokens[i]
        pred_seq_mask = pred_mask[i]
        target_positions = target_mask[i].nonzero(as_tuple=False).squeeze(-1).tolist()
        matched_pred = torch.zeros_like(pred_seq_mask, dtype=torch.bool)

        total_reference += len(target_positions)
        if not target_positions:
            continue

        for target_pos in target_positions:
            token_value = target_seq[target_pos]
            start = max(0, target_pos - tolerance)
            end = min(pred_seq.shape[0] - 1, target_pos + tolerance)

            best_match_pos = None
            best_distance = None
            for pred_pos in range(start, end + 1):
                if not pred_seq_mask[pred_pos]:
                    continue
                if matched_pred[pred_pos]:
                    continue
                if token_matcher is None:
                    is_match = pred_seq[pred_pos] == token_value
                else:
                    is_match = token_matcher(pred_seq[pred_pos], token_value)

                if not is_match:
                    continue

                distance = abs(pred_pos - target_pos)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match_pos = pred_pos

            if best_match_pos is not None:
                matched_pred[best_match_pos] = True
                total_matches += 1

    if total_reference == 0:
        return float('nan'), 0
    return total_matches / total_reference, total_reference


def make_abs_token_matcher(tolerance_steps: int):
    def matcher(pred_token: torch.Tensor, target_token: torch.Tensor) -> bool:
        return abs(int(pred_token.item()) - int(target_token.item())) <= tolerance_steps

    return matcher


def make_percent_token_matcher(token_min: int, token_max: int, tolerance_percent: float):
    token_span = max(token_max - token_min, 1)
    tolerance_tokens = int(round(token_span * tolerance_percent / 100.0))

    def matcher(pred_token: torch.Tensor, target_token: torch.Tensor) -> bool:
        return abs(int(pred_token.item()) - int(target_token.item())) <= tolerance_tokens

    return matcher


def make_category_aware_matcher(hparams: SimpleNamespace):
    pitch_matcher = make_abs_token_matcher(pitch_tolerance_steps)
    velocity_matcher = make_percent_token_matcher(
        hparams.min_vel_token, hparams.max_vel_token, velocity_tolerance_percent
    )
    timing_matcher = make_percent_token_matcher(
        hparams.min_timing_token, hparams.max_timing_token, micro_timing_tolerance_percent
    )

    def matcher(pred_token: torch.Tensor, target_token: torch.Tensor) -> bool:
        target_value = int(target_token.item())
        if hparams.min_pitch_token <= target_value < hparams.max_pitch_token:
            return pitch_matcher(pred_token, target_token)
        if hparams.min_vel_token <= target_value < hparams.max_vel_token:
            return velocity_matcher(pred_token, target_token)
        if hparams.min_timing_token <= target_value < hparams.max_timing_token:
            return timing_matcher(pred_token, target_token)
        return int(pred_token.item()) == target_value

    return matcher


def evaluate_reconstruction(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    hparams: SimpleNamespace,
    tolerance: int,
) -> dict:
    min_len = min(pred_tokens.shape[1], target_tokens.shape[1])
    pred_tokens = pred_tokens[:, :min_len]
    target_tokens = target_tokens[:, :min_len]

    target_non_pad_mask = target_tokens != hparams.pad_token
    pred_non_pad_mask = pred_tokens != hparams.pad_token

    target_pitch_mask = (target_tokens >= hparams.min_pitch_token) & (target_tokens < hparams.max_pitch_token)
    pred_pitch_mask = (pred_tokens >= hparams.min_pitch_token) & (pred_tokens < hparams.max_pitch_token)

    target_velocity_mask = (target_tokens >= hparams.min_vel_token) & (target_tokens < hparams.max_vel_token)
    pred_velocity_mask = (pred_tokens >= hparams.min_vel_token) & (pred_tokens < hparams.max_vel_token)

    target_timing_mask = (target_tokens >= hparams.min_timing_token) & (target_tokens < hparams.max_timing_token)
    pred_timing_mask = (pred_tokens >= hparams.min_timing_token) & (pred_tokens < hparams.max_timing_token)

    category_matcher = make_category_aware_matcher(hparams)

    overall_acc, overall_count = windowed_position_accuracy(
        pred_tokens,
        target_tokens,
        pred_non_pad_mask,
        target_non_pad_mask,
        tolerance,
        token_matcher=category_matcher,
    )
    pitch_acc, pitch_count = windowed_position_accuracy(
        pred_tokens, target_tokens, pred_pitch_mask, target_pitch_mask, tolerance, token_matcher=category_matcher
    )
    velocity_matcher = make_percent_token_matcher(
        hparams.min_vel_token, hparams.max_vel_token, velocity_tolerance_percent
    )
    velocity_acc, velocity_count = windowed_position_accuracy(
        pred_tokens,
        target_tokens,
        pred_velocity_mask,
        target_velocity_mask,
        tolerance,
        token_matcher=velocity_matcher,
    )
    timing_matcher = make_percent_token_matcher(
        hparams.min_timing_token, hparams.max_timing_token, micro_timing_tolerance_percent
    )
    timing_acc, timing_count = windowed_position_accuracy(
        pred_tokens,
        target_tokens,
        pred_timing_mask,
        target_timing_mask,
        tolerance,
        token_matcher=timing_matcher,
    )

    return {
        'seq_len_compared': min_len,
        'overall_token_accuracy': overall_acc,
        'overall_token_count': overall_count,
        'pitch_accuracy': pitch_acc,
        'pitch_count': pitch_count,
        'velocity_accuracy': velocity_acc,
        'velocity_count': velocity_count,
        'micro_timing_accuracy': timing_acc,
        'micro_timing_count': timing_count,
    }


##########################
# Main flow
##########################
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(__file__))
    safe_branch_name = get_safe_branch_name(project_root)

    selected_model = resolve_model_path(project_root, safe_branch_name, model_name)
    dataset_path = resolve_dataset_path(project_root, safe_branch_name, dataset_name)

    print(f'Branch: {safe_branch_name}')
    print(f'Loading model: {selected_model}')
    print(f'Loading dataset: {dataset_path}')
    print(f'Position tolerance: +/-{position_tolerance}')
    print(f'Pitch tolerance: +/-{pitch_tolerance_steps} pitch steps')
    print(f'Velocity tolerance: +/-{velocity_tolerance_percent}% of velocity token range')
    print(f'Micro timing tolerance: +/-{micro_timing_tolerance_percent}% of timing token range')

    checkpoint = torch.load(selected_model, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    hparams_dict = checkpoint['hparams']
    hparams = SimpleNamespace(**hparams_dict)

    if ONNX:
        model = ONNXWrapper(hparams)
        model.eval()
    else:
        model = TransformerVAE(hparams)
        model.load_state_dict(model_state_dict)
        model.eval()

    npz_data = np.load(dataset_path)
    if 'data' in npz_data:
        all_data = npz_data['data']
    elif 'tensor' in npz_data:
        all_data = npz_data['tensor']
    else:
        raise ValueError(f"Dataset {dataset_path} does not contain 'data' or 'tensor' key")

    if sample_skip < 0:
        sample_skip = 0
    if sample_take <= 0:
        sample_data = all_data[sample_skip:]
    else:
        sample_data = all_data[sample_skip:sample_skip + sample_take]

    if len(sample_data) == 0:
        raise ValueError('No samples selected. Adjust sample_skip/sample_take.')

    print(f'Selected samples: {len(sample_data)}')
    data_tensors = torch.tensor(sample_data, dtype=torch.long)

    with torch.no_grad():
        _, mu, logvar = model(data_tensors)
        z = reparameterize(mu, logvar)
        pred_tokens = model.sample(
            z=z,
            max_len=hparams.max_seq_len,
            temperature=0.1,
            sample=False,
        )

    metrics = evaluate_reconstruction(pred_tokens, data_tensors, hparams, position_tolerance)

    print('\nReconstruction Evaluation')
    print(f"Compared sequence length: {metrics['seq_len_compared']}")
    print(
        f"Token accuracy (non-pad): {metrics['overall_token_accuracy']:.4f} "
        f"(n={metrics['overall_token_count']})"
    )
    print(f"Pitch accuracy: {metrics['pitch_accuracy']:.4f} (n={metrics['pitch_count']})")
    print(f"Velocity accuracy: {metrics['velocity_accuracy']:.4f} (n={metrics['velocity_count']})")
    print(
        f"Micro timing accuracy: {metrics['micro_timing_accuracy']:.4f} "
        f"(n={metrics['micro_timing_count']})"
    )
