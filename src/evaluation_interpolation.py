import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from modules.interpolation import interpolate_latent_vectors_spherical
from modules.utilities import load_model
from modules.branch_paths import latest_in_branch


INTERPOLATION_STEPS = 10
NUM_RANDOM_PAIRS = 30
TAKE = 10000
SKIP = 0
RNG_SEED = 42


model, hparams = load_model(0, checkpoint_kind='best_train')


@torch.no_grad()
def _align_generated_to_target(generated: torch.Tensor, target_len: int, pad_token: int) -> torch.Tensor:
    if generated.size(0) > target_len:
        return generated[:target_len]
    if generated.size(0) < target_len:
        pad = torch.full(
            (target_len - generated.size(0),),
            int(pad_token),
            dtype=generated.dtype,
            device=generated.device,
        )
        return torch.cat([generated, pad], dim=0)
    return generated


@torch.no_grad()
def _token_similarity(anchor_tokens: torch.Tensor, generated_tokens: torch.Tensor, pad_token: int) -> float:
    aligned = _align_generated_to_target(generated_tokens, anchor_tokens.size(0), pad_token)
    valid_mask = anchor_tokens != int(pad_token)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return 0.0
    matches = int(((anchor_tokens == aligned) & valid_mask).sum().item())
    return float(matches / valid_count)


@torch.no_grad()
def _evaluate_pair(data: np.ndarray, idx_a: int, idx_b: int) -> dict:
    pair = np.array([data[idx_a], data[idx_b]])
    anchors = torch.tensor(pair, dtype=torch.long)

    _logits, mu, _logvar = model(anchors)
    z = mu  # deterministic path, same as inference.py

    interpolated_vectors = interpolate_latent_vectors_spherical(z, steps_between=INTERPOLATION_STEPS)
    stack = torch.stack([vec.squeeze(0) for vec in interpolated_vectors], dim=0).float()

    preds = model.sample(z=stack, max_len=hparams.max_seq_len, temperature=0.1, sample=False)

    pad_token = int(hparams.pad_token)
    sim_to_a0 = []
    sim_to_a1 = []
    for i in range(preds.size(0)):
        sim_to_a0.append(_token_similarity(anchors[0], preds[i], pad_token))
        sim_to_a1.append(_token_similarity(anchors[1], preds[i], pad_token))

    sim_to_a0 = np.asarray(sim_to_a0, dtype=np.float32)
    sim_to_a1 = np.asarray(sim_to_a1, dtype=np.float32)

    diff = sim_to_a1 - sim_to_a0
    x = np.arange(len(diff), dtype=np.float32)
    slope = float(np.polyfit(x, diff, 1)[0]) if len(diff) > 1 else 0.0

    monotonic_fraction = float(np.mean(np.diff(diff) >= 0.0)) if len(diff) > 2 else 0.0

    return {
        'sim_to_a0': sim_to_a0,
        'sim_to_a1': sim_to_a1,
        'mean_sim_a0': float(np.mean(sim_to_a0)),
        'mean_sim_a1': float(np.mean(sim_to_a1)),
        'start_a0_advantage': float(sim_to_a0[0] - sim_to_a1[0]),
        'end_a1_advantage': float(sim_to_a1[-1] - sim_to_a0[-1]),
        'trend_slope': slope,
        'monotonic_fraction': monotonic_fraction,
    }


def _plot_similarity_table(sim_to_a0_avg: np.ndarray, sim_to_a1_avg: np.ndarray) -> Path:
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'interpolation_similarity_plot.png'

    x = np.arange(len(sim_to_a0_avg))
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, sim_to_a0_avg, marker='o', linewidth=1.8, label='Similarity->A0')
    plt.plot(x, sim_to_a1_avg, marker='o', linewidth=1.8, label='Similarity->A1')
    plt.xlabel('Interpolation Index')
    plt.ylabel('Average Similarity')
    plt.title('Interpolation Similarity by Index (Averaged over Random Pairs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


@torch.no_grad()
def main() -> None:
    latest_dataset = latest_in_branch('data', 'dataset-groove.npz')
    if latest_dataset is None:
        raise FileNotFoundError('No dataset files found in data/.')

    print(f'Loading dataset: {latest_dataset}')
    data = np.load(latest_dataset)['data']
    print(f'Using samples slice: [{SKIP}:{SKIP + TAKE}]')
    data = data[SKIP:SKIP + TAKE]

    if len(data) < 2:
        raise ValueError('Dataset needs at least two sequences.')

    model.eval()

    rng = np.random.default_rng(RNG_SEED)
    pair_metrics = []

    print('')
    print('Starting interpolation evaluation...')
    print(f'Interpolation steps: {INTERPOLATION_STEPS}')
    print(f'Random pairs: {NUM_RANDOM_PAIRS}')

    for pair_idx in range(NUM_RANDOM_PAIRS):
        idx_a, idx_b = rng.choice(len(data), size=2, replace=False)
        print(f'[{pair_idx + 1}/{NUM_RANDOM_PAIRS}] Evaluating pair ({idx_a}, {idx_b})...')

        metrics = _evaluate_pair(data, int(idx_a), int(idx_b))
        pair_metrics.append(metrics)

        print(
            f"[{pair_idx + 1}/{NUM_RANDOM_PAIRS}] "
            f"slope={metrics['trend_slope']:.6f}, "
            f"start_a0_adv={metrics['start_a0_advantage']:.6f}, "
            f"end_a1_adv={metrics['end_a1_advantage']:.6f}"
        )

    def avg(key: str) -> float:
        return float(np.mean([m[key] for m in pair_metrics]))

    def std(key: str) -> float:
        return float(np.std([m[key] for m in pair_metrics]))

    sim_to_a0_all = np.stack([m['sim_to_a0'] for m in pair_metrics], axis=0)
    sim_to_a1_all = np.stack([m['sim_to_a1'] for m in pair_metrics], axis=0)
    sim_to_a0_avg = np.mean(sim_to_a0_all, axis=0)
    sim_to_a1_avg = np.mean(sim_to_a1_all, axis=0)

    print('')
    print('===' * 10)
    print('Averaged Interpolation Quality Metrics (30 random pairs)')
    print(f"mean_sim_a0        : {avg('mean_sim_a0'):.6f} +- {std('mean_sim_a0'):.6f}")
    print(f"mean_sim_a1        : {avg('mean_sim_a1'):.6f} +- {std('mean_sim_a1'):.6f}")
    print(f"start_a0_advantage : {avg('start_a0_advantage'):.6f} +- {std('start_a0_advantage'):.6f}")
    print(f"end_a1_advantage   : {avg('end_a1_advantage'):.6f} +- {std('end_a1_advantage'):.6f}")
    print(f"trend_slope        : {avg('trend_slope'):.6f} +- {std('trend_slope'):.6f}")
    print(f"monotonic_fraction : {avg('monotonic_fraction'):.6f} +- {std('monotonic_fraction'):.6f}")
    print('===' * 10)

    print('')
    print('Idx | Similarity->A0 | Similarity->A1 | Closer Anchor')
    for idx, (sim0, sim1) in enumerate(zip(sim_to_a0_avg, sim_to_a1_avg)):
        closer = 'A0' if sim0 >= sim1 else 'A1'
        print(f'{idx:>3} | {sim0:>14.6f} | {sim1:>14.6f} | {closer}')

    plot_path = _plot_similarity_table(sim_to_a0_avg, sim_to_a1_avg)
    print('')
    print(f'Saved similarity plot: {plot_path}')


if __name__ == '__main__':
    main()
