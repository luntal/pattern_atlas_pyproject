import glob
import os
import subprocess
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules.interpolation import interpolate_latent_vectors_spherical
from modules.model import ONNXWrapper, TransformerVAE

"""
Evaluate interpolation quality by comparing each interpolated output sequence
against the two source (original) sequences of each interpolation pair.
"""


##########################
# Configuration
##########################
ONNX = False
# model_name = None  # take the latest model in the branch
model_name = 'model_2026-04-08_09-52-22.pt'
dataset_name = 'dataset-groove.npz'
sample_skip = 0
sample_take = 0  # <= 0 means use all samples after skip

# Interpolation settings
n_interpolation_pairs = 20
interpolation_steps = 10  # number of vectors between each pair endpoint
pair_sampling_seed = 42
sampling_temperature = 0.1
position_tolerance = 6  # Match token within +/- this many positions.
pitch_tolerance_steps = 1
velocity_tolerance_percent = 30
micro_timing_tolerance_percent = 20

# Output
plot_filename = 'evaluation_interpolation.png'


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


def load_model(onnx_mode: bool, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    hparams_dict = checkpoint['hparams']
    hparams = SimpleNamespace(**hparams_dict)

    if onnx_mode:
        model = ONNXWrapper(hparams)
        model.eval()
    else:
        model = TransformerVAE(hparams)
        model.load_state_dict(model_state_dict)
        model.eval()

    return model, hparams


def load_dataset(dataset_path: str) -> np.ndarray:
    npz_data = np.load(dataset_path)
    if 'data' in npz_data:
        return npz_data['data']
    if 'tensor' in npz_data:
        return npz_data['tensor']
    raise ValueError(f"Dataset {dataset_path} does not contain 'data' or 'tensor' key")


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


def evaluate_windowed_metrics(
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

    overall_acc, _ = windowed_position_accuracy(
        pred_tokens,
        target_tokens,
        pred_non_pad_mask,
        target_non_pad_mask,
        tolerance,
        token_matcher=category_matcher,
    )
    pitch_acc, _ = windowed_position_accuracy(
        pred_tokens, target_tokens, pred_pitch_mask, target_pitch_mask, tolerance, token_matcher=category_matcher
    )
    velocity_acc, _ = windowed_position_accuracy(
        pred_tokens,
        target_tokens,
        pred_velocity_mask,
        target_velocity_mask,
        tolerance,
        token_matcher=category_matcher,
    )
    timing_acc, _ = windowed_position_accuracy(
        pred_tokens,
        target_tokens,
        pred_timing_mask,
        target_timing_mask,
        tolerance,
        token_matcher=category_matcher,
    )

    return {
        'overall': overall_acc,
        'pitch': pitch_acc,
        'velocity': velocity_acc,
        'timing': timing_acc,
    }


def select_interpolation_pairs(n_samples: int, n_pairs: int, seed: int) -> list[tuple[int, int]]:
    if n_samples < 2:
        raise ValueError('Need at least 2 samples to build interpolation pairs.')

    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_samples, size=2, replace=False)
        pairs.append((int(i), int(j)))
    return pairs


def build_interpolated_predictions(
    model,
    source_sequences: np.ndarray,
    max_seq_len: int,
    steps_between: int,
    temperature: float,
) -> torch.Tensor:
    source_tensors = torch.tensor(source_sequences, dtype=torch.long)

    with torch.no_grad():
        _, mu, _ = model(source_tensors)
        # Deterministic interpolation in latent space using mean vectors.
        latent_vectors = interpolate_latent_vectors_spherical(mu, steps_between=steps_between)
        latent_stack = torch.stack([vec.squeeze(0) for vec in latent_vectors], dim=0).float()
        preds = model.sample(
            z=latent_stack,
            max_len=max_seq_len,
            temperature=temperature,
            sample=False,
        )

    return preds


def save_interpolation_plot(
    save_path: str,
    mean_overall_to_a: np.ndarray,
    mean_overall_to_b: np.ndarray,
):
    x = np.arange(len(mean_overall_to_a))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(x, mean_overall_to_a, label='x vs original A', marker='o')
    ax.plot(x, mean_overall_to_b, label='x vs original B', marker='o')
    ax.set_xlabel('Interpolation position (0=start original A, last=end original B)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Interpolation Accuracy (Windowed Position Match)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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

    model, hparams = load_model(ONNX, selected_model)
    all_data = load_dataset(dataset_path)

    if sample_skip < 0:
        sample_skip = 0
    if sample_take <= 0:
        sample_data = all_data[sample_skip:]
    else:
        sample_data = all_data[sample_skip:sample_skip + sample_take]

    if len(sample_data) < 2:
        raise ValueError('Need at least 2 selected samples for interpolation evaluation.')

    pairs = select_interpolation_pairs(len(sample_data), n_interpolation_pairs, pair_sampling_seed)

    print(f'Selected samples: {len(sample_data)}')
    print(f'Interpolation pairs: {n_interpolation_pairs}')
    print(f'Interpolation steps between endpoints: {interpolation_steps}')
    print(f'Position tolerance: +/-{position_tolerance}')
    print(f'Pitch tolerance: +/-{pitch_tolerance_steps} pitch steps')
    print(f'Velocity tolerance: +/-{velocity_tolerance_percent}% of velocity token range')
    print(f'Micro timing tolerance: +/-{micro_timing_tolerance_percent}% of timing token range')

    n_positions = interpolation_steps + 2
    overall_to_a = np.zeros((n_interpolation_pairs, n_positions), dtype=float)
    overall_to_b = np.zeros((n_interpolation_pairs, n_positions), dtype=float)
    overall_best = np.zeros((n_interpolation_pairs, n_positions), dtype=float)
    pitch_best = np.zeros((n_interpolation_pairs, n_positions), dtype=float)
    velocity_best = np.zeros((n_interpolation_pairs, n_positions), dtype=float)
    timing_best = np.zeros((n_interpolation_pairs, n_positions), dtype=float)

    for pair_idx, (idx_a, idx_b) in enumerate(pairs):
        source_pair = np.array([sample_data[idx_a], sample_data[idx_b]])
        pred_tokens = build_interpolated_predictions(
            model=model,
            source_sequences=source_pair,
            max_seq_len=hparams.max_seq_len,
            steps_between=interpolation_steps,
            temperature=sampling_temperature,
        )

        ref_a = torch.tensor(source_pair[0], dtype=torch.long).unsqueeze(0)
        ref_b = torch.tensor(source_pair[1], dtype=torch.long).unsqueeze(0)

        for pos in range(n_positions):
            pred_seq = pred_tokens[pos].unsqueeze(0)
            metrics_a = evaluate_windowed_metrics(pred_seq, ref_a, hparams, position_tolerance)
            metrics_b = evaluate_windowed_metrics(pred_seq, ref_b, hparams, position_tolerance)

            overall_to_a[pair_idx, pos] = metrics_a['overall']
            overall_to_b[pair_idx, pos] = metrics_b['overall']
            overall_best[pair_idx, pos] = max(metrics_a['overall'], metrics_b['overall'])
            pitch_best[pair_idx, pos] = max(metrics_a['pitch'], metrics_b['pitch'])
            velocity_best[pair_idx, pos] = max(metrics_a['velocity'], metrics_b['velocity'])
            timing_best[pair_idx, pos] = max(metrics_a['timing'], metrics_b['timing'])

        mean_mid = np.mean(overall_best[pair_idx, 1:-1]) if n_positions > 2 else overall_best[pair_idx, 0]
        print(
            f'Pair {pair_idx + 1}/{n_interpolation_pairs} '
            f'(sample idx {idx_a} <-> {idx_b}) | '
            f'mean best overall (intermediate only): {mean_mid:.4f}'
        )

    mean_overall_to_a = np.mean(overall_to_a, axis=0)
    mean_overall_to_b = np.mean(overall_to_b, axis=0)
    mean_overall_best = np.mean(overall_best, axis=0)
    mean_pitch_best = np.mean(pitch_best, axis=0)
    mean_velocity_best = np.mean(velocity_best, axis=0)
    mean_timing_best = np.mean(timing_best, axis=0)

    docs_branch_dir = os.path.join(project_root, 'docs', safe_branch_name)
    os.makedirs(docs_branch_dir, exist_ok=True)
    plot_path = os.path.join(docs_branch_dir, plot_filename)

    save_interpolation_plot(
        save_path=plot_path,
        mean_overall_to_a=mean_overall_to_a,
        mean_overall_to_b=mean_overall_to_b,
    )

    print('\nInterpolation Evaluation Summary')
    print(f'Mean overall vs original A (all positions): {np.mean(mean_overall_to_a):.4f}')
    print(f'Mean overall vs original B (all positions): {np.mean(mean_overall_to_b):.4f}')
    print(f'Mean overall vs closer original (all positions): {np.mean(mean_overall_best):.4f}')
    print(f'Mean pitch vs closer original (all positions): {np.mean(mean_pitch_best):.4f}')
    print(f'Mean velocity vs closer original (all positions): {np.mean(mean_velocity_best):.4f}')
    print(f'Mean micro timing vs closer original (all positions): {np.mean(mean_timing_best):.4f}')
    print(f'Matplotlib output saved to: {plot_path}')
