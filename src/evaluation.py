from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modules.model import TransformerVAE, make_beat_tokens
from modules.radial_basis_function import compute_rbf_weights, rbf_interpolate_exact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "dev" / "model_2026-04-08_09-52-22.pt"
DATASET_PATH = PROJECT_ROOT / "data" / "exp" / "dataset-merged.npz"

ANCHOR_COUNT = 5
GRID_RESOLUTION = 20
FIELD_HALF_EXTENT = 2.0
RBF_SIGMA = 0.30


@dataclass
class RhythmStats:
	mean_pitch: float
	mean_velocity: float
	mean_microtiming: float


def _center_timing_token(hparams: SimpleNamespace) -> int:
	if hasattr(hparams, "center_timing_token"):
		return int(hparams.center_timing_token)
	if hasattr(hparams, "min_timing_token") and hasattr(hparams, "n_timing"):
		return int(hparams.min_timing_token) + (int(hparams.n_timing) // 2)
	if hasattr(hparams, "min_timing_token") and hasattr(hparams, "max_timing_token"):
		span = int(hparams.max_timing_token) - int(hparams.min_timing_token)
		return int(hparams.min_timing_token) + (span // 2)
	raise AttributeError("Missing timing token configuration in hparams.")


def _select_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def _load_model_from_path(model_path: Path, device: torch.device) -> tuple[TransformerVAE, SimpleNamespace]:
	if not model_path.exists():
		raise FileNotFoundError(f"Model not found: {model_path}")

	checkpoint = torch.load(str(model_path), map_location="cpu")
	hparams = SimpleNamespace(**checkpoint["hparams"])
	model = TransformerVAE(hparams)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()
	return model, hparams


def _load_validation_split_like_train(hparams: SimpleNamespace, dataset_path: Path = DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset not found: {dataset_path}")

	full_data = np.load(str(dataset_path), allow_pickle=True)
	data = full_data["data"]
	metadata = full_data["metadata"]

	# Keep train.py behavior: optional truncation using n_samples.
	n_samples = int(getattr(hparams, "n_samples", 0))
	if n_samples > 0 and n_samples < data.shape[0]:
		data = data[:n_samples]
		metadata = metadata[:n_samples]

	# Keep train.py behavior: deterministic shuffle by seed before split.
	if data.shape[0] > 1:
		rng = np.random.default_rng(int(getattr(hparams, "seed", 42)))
		perm = rng.permutation(data.shape[0])
		data = data[perm]
		metadata = metadata[perm]

	validation_split = float(getattr(hparams, "validation_split", 0.1))
	val_size = int(len(data) * validation_split)
	train_size = len(data) - val_size

	val_data = data[train_size:]
	val_metadata = metadata[train_size:]

	if len(val_data) == 0:
		raise ValueError("Validation split is empty. Increase dataset size or validation_split.")

	return val_data, val_metadata


def _build_decoder_inputs(tokens: torch.Tensor, hparams: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	batch_size = tokens.size(0)
	sos = torch.full((batch_size, 1), int(hparams.sos_token), dtype=torch.long, device=tokens.device)
	decoder_input = torch.cat([sos, tokens[:, :-1]], dim=1)
	decoder_input[decoder_input == int(hparams.eos_token)] = int(hparams.pad_token)

	beat_tokens_enc = make_beat_tokens(
		tokens,
		step_token=int(hparams.step_token),
		pad_token=int(hparams.pad_token),
		total_steps=int(hparams.total_steps),
	)
	beat0 = torch.zeros((batch_size, 1), dtype=torch.long, device=tokens.device)
	beat_tokens_dec = torch.cat([beat0, beat_tokens_enc[:, :-1]], dim=1)
	return decoder_input, beat_tokens_enc, beat_tokens_dec


def _trim_sequence(tokens: np.ndarray, hparams: SimpleNamespace) -> np.ndarray:
	seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
	out = []
	for tok in seq.tolist():
		t = int(tok)
		if t == int(hparams.pad_token):
			break
		out.append(t)
		if t == int(hparams.eos_token):
			break
	if not out or out[-1] != int(hparams.eos_token):
		out.append(int(hparams.eos_token))
	return np.asarray(out, dtype=np.int64)


def _decode_pitch_vel_timing_events(tokens: np.ndarray, hparams: SimpleNamespace) -> list[tuple[int, int, int, int]]:
	events: list[tuple[int, int, int, int]] = []
	seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
	step_idx = 0
	i = 0
	while i < len(seq):
		tok = int(seq[i])
		if tok == int(hparams.eos_token) or tok == int(hparams.pad_token):
			break
		if tok == int(hparams.step_token):
			step_idx += 1
			i += 1
			continue

		if i + 2 < len(seq):
			t0 = int(seq[i])
			t1 = int(seq[i + 1])
			t2 = int(seq[i + 2])

			is_pitch = int(hparams.min_pitch_token) <= t0 < int(hparams.max_pitch_token)
			is_vel = int(hparams.min_vel_token) <= t1 < int(hparams.max_vel_token)
			is_timing = int(hparams.min_timing_token) <= t2 < int(hparams.max_timing_token)

			if is_pitch and is_vel and is_timing:
				pitch = t0 - int(hparams.min_pitch_token)
				velocity = t1 - int(hparams.min_vel_token)
				microtiming = t2 - _center_timing_token(hparams)
				events.append((step_idx, pitch, velocity, microtiming))
				i += 3
				continue

		i += 1

	return events


def _rhythm_stats(tokens: np.ndarray, hparams: SimpleNamespace) -> RhythmStats:
	events = _decode_pitch_vel_timing_events(tokens, hparams)
	if len(events) == 0:
		return RhythmStats(np.nan, np.nan, np.nan)

	arr = np.asarray(events, dtype=np.float32)
	mean_pitch = float(np.mean(arr[:, 1]))
	mean_velocity = float(np.mean(arr[:, 2]))
	mean_microtiming = float(np.mean(arr[:, 3]))
	return RhythmStats(mean_pitch, mean_velocity, mean_microtiming)


def _decode_latent_greedy(model: TransformerVAE, hparams: SimpleNamespace, latent_vector: np.ndarray) -> np.ndarray:
	device = next(model.parameters()).device
	z = torch.from_numpy(np.asarray(latent_vector, dtype=np.float32)).to(device).unsqueeze(0)

	decoder = model.decoder
	memory = decoder.z_to_mem(z).view(1, decoder.n_memory_tokens, decoder.emb_size)
	max_beat_index = decoder.beat_embedding.num_embeddings - 1

	cur_tokens = torch.full((1, 1), int(hparams.sos_token), dtype=torch.long, device=device)
	beat_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
	generated: list[int] = []

	with torch.no_grad():
		for _ in range(int(hparams.max_seq_len)):
			t = cur_tokens.shape[1]
			if t > decoder.max_seq_len:
				break

			emb = decoder.token_embedding(cur_tokens)
			beat_emb = decoder.beat_embedding(beat_tokens)
			emb = decoder.projection(torch.cat((emb, beat_emb), dim=-1))
			pos_emb = decoder.pos_embedding(torch.arange(t, device=device)).unsqueeze(0)
			emb = emb + pos_emb

			tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t).to(device)
			tgt_key_padding_mask = (cur_tokens == decoder.pad_token).float()

			out = decoder.transformer_decoder(
				tgt=emb,
				memory=memory,
				tgt_mask=tgt_mask,
				tgt_key_padding_mask=tgt_key_padding_mask,
			)
			logits = decoder.fc_out(out)
			next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
			token = int(next_tok.item())
			generated.append(token)

			cur_tokens = torch.cat([cur_tokens, next_tok], dim=1)
			is_step = (cur_tokens == int(hparams.step_token)).to(torch.long)
			beat_tokens = torch.cumsum(is_step, dim=1)
			beat_tokens += 1
			beat_tokens[:, 0] = 0
			beat_tokens = beat_tokens.clamp(min=0, max=max_beat_index)

			if token == int(hparams.eos_token):
				break

	return np.asarray(generated, dtype=np.int64)


def evaluate_reconstruction(model: TransformerVAE, hparams: SimpleNamespace, val_data: np.ndarray) -> dict:
	device = next(model.parameters()).device
	dataset = TensorDataset(torch.from_numpy(val_data).to(torch.long))
	loader = DataLoader(dataset, batch_size=int(hparams.batch_size), shuffle=False)

	ce_sum = 0.0
	token_count = 0
	token_correct = 0
	exact_matches = 0
	n_sequences = 0

	pitch_correct = 0
	pitch_total = 0
	velocity_abs_err_sum = 0.0
	velocity_total = 0
	timing_abs_err_sum = 0.0
	timing_total = 0

	model.eval()
	with torch.no_grad():
		for (tokens_cpu,) in loader:
			tokens = tokens_cpu.to(device)
			decoder_input, beat_tokens_enc, beat_tokens_dec = _build_decoder_inputs(tokens, hparams)

			mu, _ = model.encoder(tokens, beat_tokens_enc)
			logits = model.decoder(decoder_input, mu, beat_tokens_dec)
			preds = torch.argmax(logits, dim=-1)

			valid_mask = tokens != int(hparams.pad_token)
			ce_batch = F.cross_entropy(
				logits.reshape(-1, logits.size(-1)),
				tokens.reshape(-1),
				ignore_index=int(hparams.pad_token),
				reduction="sum",
			)
			ce_sum += float(ce_batch.item())

			token_count += int(valid_mask.sum().item())
			token_correct += int(((preds == tokens) & valid_mask).sum().item())

			seq_exact = ((preds == tokens) | (~valid_mask)).all(dim=1)
			exact_matches += int(seq_exact.sum().item())
			n_sequences += int(tokens.size(0))

			pitch_mask = (tokens >= int(hparams.min_pitch_token)) & (tokens < int(hparams.max_pitch_token))
			velocity_mask = (tokens >= int(hparams.min_vel_token)) & (tokens < int(hparams.max_vel_token))
			timing_mask = (tokens >= int(hparams.min_timing_token)) & (tokens < int(hparams.max_timing_token))

			pitch_correct += int(((preds == tokens) & pitch_mask).sum().item())
			pitch_total += int(pitch_mask.sum().item())

			velocity_abs_err_sum += float(torch.abs(preds[velocity_mask] - tokens[velocity_mask]).float().sum().item())
			velocity_total += int(velocity_mask.sum().item())

			timing_abs_err_sum += float(torch.abs(preds[timing_mask] - tokens[timing_mask]).float().sum().item())
			timing_total += int(timing_mask.sum().item())

	reconstruction_loss = ce_sum / max(1, token_count)
	token_accuracy = token_correct / max(1, token_count)
	exact_match_percent = 100.0 * exact_matches / max(1, n_sequences)
	pitch_accuracy = pitch_correct / max(1, pitch_total)
	velocity_mae = velocity_abs_err_sum / max(1, velocity_total)
	microtiming_mae = timing_abs_err_sum / max(1, timing_total)

	return {
		"n_validation_sequences": n_sequences,
		"exact_match_percent": exact_match_percent,
		"token_accuracy": token_accuracy,
		"cross_entropy_reconstruction_loss": reconstruction_loss,
		"pitch_accuracy": pitch_accuracy,
		"velocity_mae": velocity_mae,
		"microtiming_mae": microtiming_mae,
	}


def _make_circular_control_points(n_points: int, field_half_extent: float = FIELD_HALF_EXTENT) -> np.ndarray:
	radius = 0.75 * float(field_half_extent)
	angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
	return np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius]).astype(np.float32)


def _similarity_heatmaps(
	model: TransformerVAE,
	hparams: SimpleNamespace,
	anchor_sequences: np.ndarray,
	grid_resolution: int = GRID_RESOLUTION,
	field_half_extent: float = FIELD_HALF_EXTENT,
	sigma: float = RBF_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[RhythmStats]]:
	device = next(model.parameters()).device
	anchors = torch.from_numpy(anchor_sequences).to(device=device, dtype=torch.long)

	with torch.no_grad():
		beat_tokens = make_beat_tokens(
			anchors,
			step_token=int(hparams.step_token),
			pad_token=int(hparams.pad_token),
			total_steps=int(hparams.total_steps),
		)
		mu, _ = model.encoder(anchors, beat_tokens)
		anchor_latents = mu.detach().cpu().numpy().astype(np.float32)

	control_points = _make_circular_control_points(len(anchor_latents), field_half_extent=field_half_extent)
	weights = compute_rbf_weights(control_points, anchor_latents, sigma=sigma)

	anchor_stats = [_rhythm_stats(_trim_sequence(seq, hparams), hparams) for seq in anchor_sequences]

	x = np.linspace(-field_half_extent, field_half_extent, int(grid_resolution))
	y = np.linspace(-field_half_extent, field_half_extent, int(grid_resolution))

	pitch_heatmap = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
	velocity_heatmap = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
	timing_heatmap = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

	for row, yy in enumerate(y):
		for col, xx in enumerate(x):
			cursor = np.asarray([xx, yy], dtype=np.float32)
			latent = rbf_interpolate_exact(control_points, weights, cursor, sigma=sigma).astype(np.float32)
			decoded_tokens = _decode_latent_greedy(model, hparams, latent)
			decoded_stats = _rhythm_stats(decoded_tokens, hparams)

			dp = []
			dv = []
			dt = []
			for anchor in anchor_stats:
				if np.isnan(decoded_stats.mean_pitch) or np.isnan(anchor.mean_pitch):
					continue
				dp.append(abs(decoded_stats.mean_pitch - anchor.mean_pitch))
				dv.append(abs(decoded_stats.mean_velocity - anchor.mean_velocity))
				dt.append(abs(decoded_stats.mean_microtiming - anchor.mean_microtiming))

			pitch_heatmap[row, col] = float(np.mean(dp)) if dp else np.nan
			velocity_heatmap[row, col] = float(np.mean(dv)) if dv else np.nan
			timing_heatmap[row, col] = float(np.mean(dt)) if dt else np.nan

		progress = ((row + 1) / max(1, grid_resolution)) * 100.0
		print(f"Heatmap progress: row {row + 1}/{grid_resolution} ({progress:.1f}%)")

	return control_points, pitch_heatmap, velocity_heatmap, timing_heatmap, anchor_stats


def _render_heatmap(
	data: np.ndarray,
	control_points: np.ndarray,
	out_path: Path,
	title: str,
	field_half_extent: float = FIELD_HALF_EXTENT,
):
	fig, ax = plt.subplots(figsize=(7, 6))
	image = ax.imshow(
		data,
		origin="lower",
		extent=[-field_half_extent, field_half_extent, -field_half_extent, field_half_extent],
		aspect="equal",
		cmap="viridis",
	)
	ax.scatter(control_points[:, 0], control_points[:, 1], c="white", edgecolors="black", s=80, label="Anchors")
	for idx, (x, y) in enumerate(control_points):
		ax.text(float(x) + 0.04, float(y) + 0.04, str(idx), color="white", fontsize=9)
	ax.set_title(title)
	ax.set_xlabel("Field X")
	ax.set_ylabel("Field Y")
	ax.legend(loc="upper right")
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
	fig.tight_layout()
	fig.savefig(out_path, dpi=180)
	plt.close(fig)


def main():
	device = _select_device()
	model, hparams = _load_model_from_path(MODEL_PATH, device)
	val_data, val_metadata = _load_validation_split_like_train(hparams, dataset_path=DATASET_PATH)

	print("=" * 60)
	print(f"Device: {device}")
	print(f"Model: {MODEL_PATH}")
	print(f"Validation samples: {len(val_data)}")
	print("=" * 60)

	reconstruction_metrics = evaluate_reconstruction(model, hparams, val_data)

	anchor_count = min(int(ANCHOR_COUNT), int(len(val_data)))
	anchor_sequences = val_data[:anchor_count]
	anchor_metadata = val_metadata[:anchor_count]

	control_points, pitch_hm, velocity_hm, timing_hm, anchor_stats = _similarity_heatmaps(
		model,
		hparams,
		anchor_sequences,
		grid_resolution=GRID_RESOLUTION,
		field_half_extent=FIELD_HALF_EXTENT,
		sigma=RBF_SIGMA,
	)

	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	out_dir = PROJECT_ROOT / "outputs" / "evaluation"
	out_dir.mkdir(parents=True, exist_ok=True)

	pitch_png = out_dir / f"heatmap_pitch_distance_{timestamp}.png"
	velocity_png = out_dir / f"heatmap_velocity_distance_{timestamp}.png"
	timing_png = out_dir / f"heatmap_microtiming_distance_{timestamp}.png"

	_render_heatmap(pitch_hm, control_points, pitch_png, "Average Pitch Distance to Anchor Rhythms")
	_render_heatmap(velocity_hm, control_points, velocity_png, "Average Velocity Distance to Anchor Rhythms")
	_render_heatmap(timing_hm, control_points, timing_png, "Average Microtiming Distance to Anchor Rhythms")

	report = {
		"model_path": str(MODEL_PATH),
		"dataset_path": str(DATASET_PATH),
		"dataset_name": str(hparams.dataset_name),
		"validation_size": int(len(val_data)),
		"reconstruction_metrics": reconstruction_metrics,
		"interpolation": {
			"anchor_count": int(anchor_count),
			"grid_resolution": int(GRID_RESOLUTION),
			"field_half_extent": float(FIELD_HALF_EXTENT),
			"rbf_sigma": float(RBF_SIGMA),
			"control_points": control_points.tolist(),
			"anchors": [str(item[0]) if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0 else str(item)
						for item in anchor_metadata],
			"anchor_stats": [
				{
					"mean_pitch": float(s.mean_pitch) if not np.isnan(s.mean_pitch) else None,
					"mean_velocity": float(s.mean_velocity) if not np.isnan(s.mean_velocity) else None,
					"mean_microtiming": float(s.mean_microtiming) if not np.isnan(s.mean_microtiming) else None,
				}
				for s in anchor_stats
			],
			"heatmaps": {
				"pitch_distance_png": str(pitch_png),
				"velocity_distance_png": str(velocity_png),
				"microtiming_distance_png": str(timing_png),
			},
		},
	}

	report_path = out_dir / f"evaluation_report_{timestamp}.json"
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	npz_path = out_dir / f"evaluation_heatmaps_{timestamp}.npz"
	np.savez_compressed(
		npz_path,
		pitch_distance=pitch_hm,
		velocity_distance=velocity_hm,
		microtiming_distance=timing_hm,
		control_points=control_points,
	)

	print("Reconstruction Quality")
	print(f"- Exact match (% identical sequences): {reconstruction_metrics['exact_match_percent']:.3f}%")
	print(f"- Token accuracy: {reconstruction_metrics['token_accuracy']:.6f}")
	print(f"- Cross entropy / reconstruction loss: {reconstruction_metrics['cross_entropy_reconstruction_loss']:.6f}")
	print(f"- Pitch accuracy: {reconstruction_metrics['pitch_accuracy']:.6f}")
	print(f"- Velocity MAE: {reconstruction_metrics['velocity_mae']:.6f}")
	print(f"- Micro Timing MAE: {reconstruction_metrics['microtiming_mae']:.6f}")

	print("\nInterpolation Quality")
	print(f"- Anchors from validation set: {anchor_count}")
	print(f"- Grid resolution: {GRID_RESOLUTION} x {GRID_RESOLUTION}")
	print(f"- Pitch distance heatmap: {pitch_png}")
	print(f"- Velocity distance heatmap: {velocity_png}")
	print(f"- Microtiming distance heatmap: {timing_png}")

	print("\nArtifacts")
	print(f"- Report JSON: {report_path}")
	print(f"- Heatmaps NPZ: {npz_path}")


if __name__ == "__main__":
	main()
