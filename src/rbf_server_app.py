from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

from modules.branch_paths import get_current_branch, safe_branch_name
from modules.radial_basis_function import compute_rbf_weights, rbf_interpolate_exact
from modules.utilities import load_dataset, load_model, pick_reproducible_samples


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    h_min = float(np.min(heatmap))
    h_max = float(np.max(heatmap))
    if h_max <= h_min:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - h_min) / (h_max - h_min)).astype(np.float32)


@dataclass
class FrontendState:
    cursor: np.ndarray
    control_points: np.ndarray
    heatmap_mode: str = "magnitude"
    influence_mode: str = "raw"
    decode_enabled: bool = False
    grid_resolution: int = 30
    sigma: float = 0.30
    latent_vector: np.ndarray | None = None
    last_frontend_payload: dict | None = None


class RealtimeDecoderManager:
    def __init__(self, decoder: torch.nn.Module, hparams):
        self.decoder = decoder
        self.hparams = hparams
        self.device = next(decoder.parameters()).device
        self.max_len = int(hparams.max_seq_len)

        self._lock = threading.RLock()
        self._current_job_id = 0
        self._cancel_event = None
        self._latest_generation = 0
        self._last_consumed_generation = -1

        cols = int(self.hparams.width * self.hparams.n_timing)
        self._latest_matrix = np.zeros((int(self.hparams.height), cols), dtype=np.float32)
        self._latest_progress_step = -1
        self._latest_status = "idle"

    def request_decode(self, latent_vector: np.ndarray, seed_matrix: np.ndarray):
        latent_vector = np.asarray(latent_vector, dtype=np.float32)
        seed_matrix = np.asarray(seed_matrix, dtype=np.float32)

        with self._lock:
            self._current_job_id += 1
            job_id = self._current_job_id
            if self._cancel_event is not None:
                self._cancel_event.set()
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._latest_status = "decoding"

        thread = threading.Thread(
            target=self._decode_job,
            args=(job_id, latent_vector, seed_matrix.copy(), cancel_event),
            daemon=True,
        )
        thread.start()

    def consume_latest_update(self):
        with self._lock:
            if self._latest_generation == self._last_consumed_generation:
                return None
            self._last_consumed_generation = self._latest_generation
            return {
                "matrix": self._latest_matrix.copy(),
                "progress_step": int(self._latest_progress_step),
                "status": self._latest_status,
            }

    def _publish_update(self, job_id: int, matrix: np.ndarray, progress_step: int, status: str):
        with self._lock:
            if job_id != self._current_job_id:
                return
            self._latest_matrix = matrix.astype(np.float32, copy=True)
            self._latest_progress_step = int(progress_step)
            self._latest_status = status
            self._latest_generation += 1

    def _write_wrapped_segment(self, matrix: np.ndarray, row: int, start: int, length: int, value: float):
        cols = matrix.shape[1]
        if cols <= 0 or length <= 0:
            return
        length = min(length, cols)
        s = start % cols
        e = s + length
        if e <= cols:
            matrix[row, s:e] = value
        else:
            first = cols - s
            matrix[row, s:s + first] = value
            matrix[row, 0:length - first] = value

    def _decode_job(self, job_id: int, latent_vector: np.ndarray, seed_matrix: np.ndarray, cancel_event: threading.Event):
        h = self.hparams
        n_timing = int(h.n_timing)
        cols = int(h.width * n_timing)
        step_token = int(h.step_token)
        eos_token = int(h.eos_token)
        sos_token = int(h.sos_token)
        min_pitch_token = int(h.min_pitch_token)
        max_pitch_token = int(h.max_pitch_token)
        min_vel_token = int(h.min_vel_token)
        max_vel_token = int(h.max_vel_token)
        min_timing_token = int(h.min_timing_token)
        max_timing_token = int(h.max_timing_token)
        center_timing_token = int(h.center_timing_token)

        partial_matrix = np.zeros_like(seed_matrix, dtype=np.float32)
        display_matrix = seed_matrix.copy()

        z = torch.from_numpy(latent_vector).to(self.device, dtype=torch.float32).unsqueeze(0)
        memory = self.decoder.z_to_mem(z).view(1, self.decoder.n_memory_tokens, self.decoder.emb_size)
        cur_tokens = torch.full((1, 1), sos_token, dtype=torch.long, device=self.device)
        beat_tokens = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        step_idx = 0
        pending_pitch = None
        pending_vel = None

        for _ in range(self.max_len):
            if cancel_event.is_set():
                self._publish_update(job_id, display_matrix, step_idx, "cancelled")
                return

            t = cur_tokens.shape[1]
            if t > self.decoder.max_seq_len:
                break

            with torch.no_grad():
                emb = self.decoder.token_embedding(cur_tokens)
                beat_emb = self.decoder.beat_embedding(beat_tokens)
                emb = self.decoder.projection(torch.cat((emb, beat_emb), dim=-1))
                pos_emb = self.decoder.pos_embedding(torch.arange(t, device=self.device)).unsqueeze(0)
                emb = emb + pos_emb
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t).to(self.device)
                tgt_key_padding_mask = (cur_tokens == self.decoder.pad_token).float()
                out = self.decoder.transformer_decoder(
                    tgt=emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                logits = self.decoder.fc_out(out)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            token = int(next_tok.item())
            cur_tokens = torch.cat([cur_tokens, next_tok], dim=1)
            is_step = (cur_tokens == step_token).to(torch.long)
            beat_tokens = torch.cumsum(is_step, dim=1)
            beat_tokens += 1
            beat_tokens[:, 0] = 0

            if token == eos_token:
                break
            if token == step_token:
                step_idx += 1
                continue

            if min_pitch_token <= token < max_pitch_token:
                pending_pitch = token
                pending_vel = None
                continue
            if pending_pitch is not None and min_vel_token <= token < max_vel_token:
                pending_vel = token
                continue
            if pending_pitch is not None and pending_vel is not None and min_timing_token <= token < max_timing_token:
                pitch_idx = pending_pitch - min_pitch_token
                timing = token - center_timing_token
                vel = (pending_vel - min_vel_token) / max(1, int(h.n_vel) - 1)
                note_len = max(1, n_timing // 2)
                start = step_idx * n_timing + timing
                self._write_wrapped_segment(partial_matrix, pitch_idx, start, note_len, float(vel))

                frontier = max(0, min(cols, step_idx * n_timing + note_len))
                display_matrix[:, :frontier] = partial_matrix[:, :frontier]
                self._publish_update(job_id, display_matrix, step_idx, "decoding")

                pending_pitch = None
                pending_vel = None

        display_matrix[:, :] = partial_matrix
        self._publish_update(job_id, display_matrix, step_idx, "done")


class CudaRBFRuntime:
    def __init__(self, previous: int = 0, checkpoint_kind: str | None = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_name = "CUDA"
        elif torch.backends.mps.is_available():
            device_name = "MPS (Metal Performance Shaders)"
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"
        print(f"Using device: {device_name} ({self.device})")
        self.model, self.hparams = load_model(previous=previous, onnx=False, checkpoint_kind=checkpoint_kind)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.decoder = self.model.decoder
        self.encoder = self.model.encoder

    def _write_wrapped_segment(self, matrix: np.ndarray, row: int, start: int, length: int, value: float):
        cols = matrix.shape[1]
        if cols <= 0 or length <= 0:
            return
        length = min(length, cols)
        s = start % cols
        e = s + length
        if e <= cols:
            matrix[row, s:e] = value
        else:
            first = cols - s
            matrix[row, s:s + first] = value
            matrix[row, 0:length - first] = value

    def decode_latent_to_matrix(self, latent_vector: np.ndarray) -> np.ndarray:
        h = self.hparams
        n_timing = int(h.n_timing)
        cols = int(h.width * n_timing)
        matrix = np.zeros((int(h.height), cols), dtype=np.float32)

        z = torch.from_numpy(latent_vector.astype(np.float32)).to(self.device).unsqueeze(0)
        memory = self.decoder.z_to_mem(z).view(1, self.decoder.n_memory_tokens, self.decoder.emb_size)

        step_token = int(h.step_token)
        eos_token = int(h.eos_token)
        sos_token = int(h.sos_token)
        min_pitch_token = int(h.min_pitch_token)
        max_pitch_token = int(h.max_pitch_token)
        min_vel_token = int(h.min_vel_token)
        max_vel_token = int(h.max_vel_token)
        min_timing_token = int(h.min_timing_token)
        max_timing_token = int(h.max_timing_token)
        center_timing_token = int(h.center_timing_token)
        note_len = max(1, n_timing // 2)

        cur_tokens = torch.full((1, 1), sos_token, dtype=torch.long, device=self.device)
        beat_tokens = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        step_idx = 0
        pending_pitch = None
        pending_vel = None

        with torch.no_grad():
            for _ in range(int(h.max_seq_len)):
                t = cur_tokens.shape[1]
                if t > self.decoder.max_seq_len:
                    break
                emb = self.decoder.token_embedding(cur_tokens)
                beat_emb = self.decoder.beat_embedding(beat_tokens)
                emb = self.decoder.projection(torch.cat((emb, beat_emb), dim=-1))
                pos_emb = self.decoder.pos_embedding(torch.arange(t, device=self.device)).unsqueeze(0)
                emb = emb + pos_emb
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(t).to(self.device)
                tgt_key_padding_mask = (cur_tokens == self.decoder.pad_token).float()
                out = self.decoder.transformer_decoder(
                    tgt=emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                logits = self.decoder.fc_out(out)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token = int(next_tok.item())

                cur_tokens = torch.cat([cur_tokens, next_tok], dim=1)
                is_step = (cur_tokens == step_token).to(torch.long)
                beat_tokens = torch.cumsum(is_step, dim=1)
                beat_tokens += 1
                beat_tokens[:, 0] = 0

                if token == eos_token:
                    break
                if token == step_token:
                    step_idx += 1
                    continue
                if min_pitch_token <= token < max_pitch_token:
                    pending_pitch = token
                    pending_vel = None
                    continue
                if pending_pitch is not None and min_vel_token <= token < max_vel_token:
                    pending_vel = token
                    continue
                if pending_pitch is not None and pending_vel is not None and min_timing_token <= token < max_timing_token:
                    pitch_idx = pending_pitch - min_pitch_token
                    timing = token - center_timing_token
                    vel = (pending_vel - min_vel_token) / max(1, int(h.n_vel) - 1)
                    start = step_idx * n_timing + timing
                    self._write_wrapped_segment(matrix, pitch_idx, start, note_len, float(vel))
                    pending_pitch = None
                    pending_vel = None

        return matrix

    def _tokens_from_matrix_buffer(self, matrix_buffer: np.ndarray) -> np.ndarray:
        h = self.hparams
        matrix = np.asarray(matrix_buffer, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("midi_buffer.matrix must be a 2D list.")
        if matrix.shape[0] != int(h.height):
            raise ValueError(f"midi_buffer.matrix row count must be {int(h.height)}.")

        n_timing = int(h.n_timing)
        width_steps = int(h.width)
        target_cols = width_steps * n_timing
        cols = matrix.shape[1]
        if cols < target_cols:
            matrix = np.pad(matrix, ((0, 0), (0, target_cols - cols)))
        elif cols > target_cols:
            matrix = matrix[:, :target_cols]

        tokens = []
        for step in range(width_steps):
            c0 = step * n_timing
            c1 = c0 + n_timing
            cell = matrix[:, c0:c1]
            for pitch in range(cell.shape[0]):
                row = cell[pitch]
                peak = float(np.max(row))
                if peak <= 0.02:
                    continue
                peak_col = int(np.argmax(row))
                timing = peak_col - (n_timing // 2)
                vel_idx = int(round(peak * max(1, int(h.n_vel) - 1)))
                vel_idx = max(0, min(int(h.n_vel) - 1, vel_idx))
                tokens.append(int(h.min_pitch_token) + pitch)
                tokens.append(int(h.min_vel_token) + vel_idx)
                tokens.append(int(h.center_timing_token) + timing)
            tokens.append(int(h.step_token))

        tokens.append(int(h.eos_token))
        return np.asarray(tokens, dtype=np.int64)

    def _prepare_token_batch(self, raw_tokens: np.ndarray) -> torch.Tensor:
        h = self.hparams
        tokens = np.asarray(raw_tokens, dtype=np.int64).reshape(-1)
        max_len = int(h.max_seq_len)
        if tokens.size > max_len:
            tokens = tokens[:max_len]
        if tokens.size < max_len:
            pad = np.full((max_len - tokens.size,), int(h.pad_token), dtype=np.int64)
            tokens = np.concatenate([tokens, pad], axis=0)
        return torch.from_numpy(tokens).to(self.device, dtype=torch.long).unsqueeze(0)

    def encode_midi_buffer_to_latent(self, midi_buffer) -> np.ndarray:
        if midi_buffer is None:
            raise ValueError("midi_buffer is missing while Encode MIDI is enabled.")
        if isinstance(midi_buffer, dict) and "tokens" in midi_buffer:
            raw_tokens = np.asarray(midi_buffer["tokens"], dtype=np.int64)
        elif isinstance(midi_buffer, dict) and "matrix" in midi_buffer:
            raw_tokens = self._tokens_from_matrix_buffer(np.asarray(midi_buffer["matrix"], dtype=np.float32))
        elif isinstance(midi_buffer, list):
            raw_tokens = np.asarray(midi_buffer, dtype=np.int64)
        else:
            raise ValueError("midi_buffer must be a token list, {tokens:[...]}, or {matrix:[[...], ...]}.")

        tokens_batch = self._prepare_token_batch(raw_tokens)
        beat_tokens = torch.cumsum((tokens_batch == int(self.hparams.step_token)).to(torch.long), dim=1)
        beat_tokens += 1
        beat_tokens[:, 0] = 0
        beat_tokens = beat_tokens.masked_fill(tokens_batch == int(self.hparams.pad_token), 0)
        with torch.no_grad():
            mu, _ = self.encoder(tokens_batch, beat_tokens)
        return mu.squeeze(0).detach().cpu().numpy().astype(np.float32)


class RBFServerController:
    MODE_LABELS = {
        "best_val": "best_val",
        "best_train": "best_train",
        "last": "last",
        "none": "None",
    }

    MODE_PREFIX = {
        "best_val": "model_best_val_",
        "best_train": "model_best_train_",
        "last": "model_last_",
        "none": "model_",
    }

    TIMESTAMP_PATTERN = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"

    @classmethod
    def _mode_files_static(cls, mode: str) -> list[Path]:
        branch_name = safe_branch_name(get_current_branch())
        model_dir = Path("models") / branch_name
        if not model_dir.exists():
            return []

        if mode == "none":
            kind_prefixes = (
                cls.MODE_PREFIX["best_val"],
                cls.MODE_PREFIX["best_train"],
                cls.MODE_PREFIX["last"],
            )
            files = [
                p for p in model_dir.glob("model_*.pt")
                if not any(p.name.startswith(prefix) for prefix in kind_prefixes)
                and re.fullmatch(rf"model_{cls.TIMESTAMP_PATTERN}\.pt", p.name)
            ]
        else:
            prefix = cls.MODE_PREFIX[mode]
            files = [
                p for p in model_dir.glob(f"{prefix}*.pt")
                if re.fullmatch(rf"{re.escape(prefix)}{cls.TIMESTAMP_PATTERN}\.pt", p.name)
            ]

        return sorted(files, key=os.path.getmtime)

    def __init__(self, sample_count: int = 10, sample_seed: int = 42):
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.selected_mode = "none"
        self.selected_previous = 0

        for candidate in ["none", "best_val", "best_train", "last"]:
            if self._mode_files_static(candidate):
                self.selected_mode = candidate
                break

        checkpoint_kind = None if self.selected_mode == "none" else self.selected_mode

        self.runtime = CudaRBFRuntime(previous=self.selected_previous, checkpoint_kind=checkpoint_kind)
        self.hparams = self.runtime.hparams
        self._lock = threading.Lock()
        self.decoder_manager = RealtimeDecoderManager(self.runtime.decoder, self.hparams)

        self.model_catalog = self._build_model_catalog()
        self._normalize_selection_to_catalog()

        self._load_latent_pool()
        control_points = self._make_initial_control_points(len(self.latent_vectors), field_half_extent=2.0)
        self.field_half_extent = 2.0
        self.weights = compute_rbf_weights(control_points, self.latent_vectors, sigma=0.30)
        self._last_weights_signature = None
        self._last_heatmap_signature = None
        self._cached_heatmap = np.zeros((30, 30), dtype=np.float32)
        self.recon_cols = int(self.hparams.width * self.hparams.n_timing)
        self.reconstruction_matrix = np.zeros((int(self.hparams.height), self.recon_cols), dtype=np.float32)
        self.decode_status = "idle"
        self.decode_progress_step = -1
        self._last_triggered_latent_signature = None
        self.state = FrontendState(
            cursor=np.zeros(2, dtype=np.float32),
            control_points=control_points,
            heatmap_mode="magnitude",
            influence_mode="raw",
            grid_resolution=30,
            sigma=0.30,
        )

    def _branch_model_dir(self) -> Path:
        branch_name = safe_branch_name(get_current_branch())
        return Path("models") / branch_name

    def _mode_files(self, mode: str) -> list[Path]:
        return self._mode_files_static(mode)

    def _selected_timestamp(self) -> str | None:
        items = self.model_catalog.get("modes", {}).get(self.selected_mode, {}).get("items", [])
        if 0 <= int(self.selected_previous) < len(items):
            return items[int(self.selected_previous)].get("timestamp")
        return None

    def _timestamp_from_file(self, mode: str, path: Path) -> str:
        stem = path.stem
        prefix = self.MODE_PREFIX[mode]
        if stem.startswith(prefix):
            return stem[len(prefix):]
        return stem

    def _normalize_selection_to_catalog(self):
        modes = self.model_catalog.get("modes", {})
        current = modes.get(self.selected_mode, {})
        if not current.get("enabled", False):
            for fallback in ["best_val", "best_train", "last", "none"]:
                if modes.get(fallback, {}).get("enabled", False):
                    self.selected_mode = fallback
                    self.selected_previous = 0
                    current = modes.get(self.selected_mode, {})
                    break

        items = current.get("items", [])
        if items:
            max_prev = len(items) - 1
            self.selected_previous = max(0, min(int(self.selected_previous), max_prev))
        else:
            self.selected_previous = 0

    def _build_model_catalog(self) -> dict:
        catalog = {"modes": {}}
        for mode in ["best_val", "best_train", "last", "none"]:
            files = self._mode_files(mode)
            items = []
            for previous, path in enumerate(reversed(files)):
                items.append({
                    "timestamp": self._timestamp_from_file(mode, path),
                    "previous": previous,
                })
            catalog["modes"][mode] = {
                "label": self.MODE_LABELS[mode],
                "enabled": len(items) > 0,
                "items": items,
            }
        return catalog

    def _load_latent_pool(self):
        data, metadata = load_dataset(self.hparams)
        data, metadata, _ = pick_reproducible_samples(
            data,
            metadata,
            sample_count=self.sample_count,
            sample_seed=self.sample_seed,
        )
        tokens = torch.from_numpy(data).to(self.runtime.device, dtype=torch.long)
        beat_tokens = torch.cumsum((tokens == int(self.hparams.step_token)).to(torch.long), dim=1)
        beat_tokens += 1
        beat_tokens[:, 0] = 0
        beat_tokens = beat_tokens.masked_fill(tokens == int(self.hparams.pad_token), 0)
        with torch.no_grad():
            mu, _ = self.runtime.encoder(tokens, beat_tokens)
        self.latent_vectors = mu.detach().cpu().numpy().astype(np.float32)

        labels = []
        for item in metadata:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
                filename = os.path.basename(str(item[0]))
                filename_no_ext = os.path.splitext(filename)[0]
                bpm = item[2] if len(item) > 2 else None
                bpm_value = _safe_float(bpm, -1.0)
                if bpm_value > 0:
                    labels.append(f"{filename_no_ext} (bpm: {bpm_value:.2f})")
                else:
                    labels.append(filename_no_ext)
            else:
                labels.append(str(item))
        self.labels = labels

    def reload_model(self, mode: str, previous: int = 0):
        if mode not in self.MODE_LABELS:
            raise ValueError(f"Unknown mode '{mode}'.")

        files = self._mode_files(mode)
        if not files:
            raise ValueError(f"No checkpoints available for mode '{mode}'.")

        previous = max(0, int(previous))
        if previous >= len(files):
            previous = len(files) - 1

        checkpoint_kind = None if mode == "none" else mode
        self.runtime = CudaRBFRuntime(previous=previous, checkpoint_kind=checkpoint_kind)
        self.hparams = self.runtime.hparams
        self.decoder_manager = RealtimeDecoderManager(self.runtime.decoder, self.hparams)
        self._load_latent_pool()

        self.selected_mode = mode
        self.selected_previous = previous
        self.model_catalog = self._build_model_catalog()
        self._normalize_selection_to_catalog()

        control_points = self._make_initial_control_points(len(self.latent_vectors), field_half_extent=self.field_half_extent)
        self.state.control_points = control_points
        self.state.cursor = np.zeros(2, dtype=np.float32)
        self.state.latent_vector = None
        self.recon_cols = int(self.hparams.width * self.hparams.n_timing)
        self.reconstruction_matrix = np.zeros((int(self.hparams.height), self.recon_cols), dtype=np.float32)
        self.decode_status = "idle"
        self.decode_progress_step = -1
        self._last_weights_signature = None
        self._last_heatmap_signature = None
        self._last_triggered_latent_signature = None
        self.weights = compute_rbf_weights(control_points, self.latent_vectors, sigma=self.state.sigma)
        self._cached_heatmap = np.zeros((self.state.grid_resolution, self.state.grid_resolution), dtype=np.float32)

    def _make_initial_control_points(self, n_points: int, field_half_extent: float) -> np.ndarray:
        if n_points <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        radius = field_half_extent * 0.75
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        return np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius]).astype(np.float32)

    def _compute_heatmap(self, control_points: np.ndarray, weights: np.ndarray, sigma: float, grid_resolution: int, mode: str) -> np.ndarray:
        res = int(max(4, min(60, grid_resolution)))
        x = np.linspace(-self.field_half_extent, self.field_half_extent, res)
        y = np.linspace(-self.field_half_extent, self.field_half_extent, res)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)

        sigma_sq = max(1e-8, float(sigma) ** 2)
        delta = grid_points[:, np.newaxis, :] - control_points[np.newaxis, :, :]
        dists_sq = np.sum(delta * delta, axis=2)
        phi = np.exp(-dists_sq / (2.0 * sigma_sq)).astype(np.float32)

        if mode == "confidence":
            heat = np.sum(phi, axis=1)
            return _normalize_heatmap(heat.reshape(res, res))

        latent_grid = phi @ weights
        heat = np.linalg.norm(latent_grid, axis=1)
        return _normalize_heatmap(heat.reshape(res, res))

    def _compute_influence_raw(self, control_points: np.ndarray, cursor: np.ndarray, sigma: float) -> np.ndarray:
        dists = np.linalg.norm(control_points - cursor.reshape(1, 2), axis=1)
        return np.exp(-(dists ** 2) / (2 * sigma ** 2)).astype(np.float32)

    def _make_weights_signature(self, points: np.ndarray, sigma: float):
        return (points.shape, np.round(points, 4).tobytes(), round(float(sigma), 4))

    def _make_heatmap_signature(self, points: np.ndarray, sigma: float, grid: int, mode: str):
        return (points.shape, np.round(points, 4).tobytes(), round(float(sigma), 4), int(grid), str(mode))

    def _refresh_decoder_update(self) -> bool:
        decode_update = self.decoder_manager.consume_latest_update()
        if decode_update is None:
            return False
        self.reconstruction_matrix = decode_update["matrix"]
        self.decode_status = decode_update["status"]
        self.decode_progress_step = int(decode_update["progress_step"])
        return True

    def _build_response(self, midi_buffer, encode_midi: bool) -> dict:
        points = self.state.control_points
        cursor = self.state.cursor
        sigma = self.state.sigma
        influence_mode = self.state.influence_mode
        influence_raw = self._compute_influence_raw(points, cursor, sigma)
        total = float(np.sum(influence_raw))
        influence_pct = influence_raw / total if total > 0.0 else np.zeros_like(influence_raw)

        return {
            "request_id": (self.state.last_frontend_payload or {}).get("client_request_id"),
            "cursor": cursor.tolist(),
            "draggable_points": points.tolist(),
            "heatmap": self._cached_heatmap.tolist(),
            "latent_vector": self.state.latent_vector.tolist() if self.state.latent_vector is not None else [],
            "midi_buffer": midi_buffer,
            "decoded_matrix": self.reconstruction_matrix.tolist(),
            "influence_raw": influence_raw.tolist(),
            "influence_percentage": influence_pct.tolist(),
            "influence_values": influence_pct.tolist() if influence_mode == "percentage" else influence_raw.tolist(),
            "button_values": {
                "heatmapMode": self.state.heatmap_mode,
                "influenceMode": self.state.influence_mode,
                "encodeMidi": encode_midi,
                "decodeEnabled": self.state.decode_enabled,
            },
            "slider_values": {
                "gridResolution": self.state.grid_resolution,
                "sigma": sigma,
            },
            "labels": self.labels,
            "cuda_ready": True,
            "field_half_extent": self.field_half_extent,
            "n_timing": int(self.hparams.n_timing),
            "recon_cols": int(self.hparams.width * self.hparams.n_timing),
            "height": int(self.hparams.height),
            "decode_status": self.decode_status,
            "decode_progress_step": int(self.decode_progress_step),
            "model_options": self.model_catalog,
            "model_selection": {
                "mode": self.selected_mode,
                "previous": int(self.selected_previous),
                "timestamp": self._selected_timestamp(),
            },
            "frontend_echo": {
                "cursor": self.state.last_frontend_payload.get("cursor") if self.state.last_frontend_payload else None,
                "draggable_points": self.state.last_frontend_payload.get("draggable_points") if self.state.last_frontend_payload else None,
                "heatmap": self.state.last_frontend_payload.get("heatmap") if self.state.last_frontend_payload else None,
                "latent_vector": self.state.last_frontend_payload.get("latent_vector") if self.state.last_frontend_payload else None,
                "midi_buffer": self.state.last_frontend_payload.get("midi_buffer") if self.state.last_frontend_payload else None,
                "influence_values": self.state.last_frontend_payload.get("influence_values") if self.state.last_frontend_payload else None,
                "button_values": self.state.last_frontend_payload.get("button_values") if self.state.last_frontend_payload else None,
                "slider_values": self.state.last_frontend_payload.get("slider_values") if self.state.last_frontend_payload else None,
            },
        }

    def snapshot_response(self) -> dict:
        with self._lock:
            self._refresh_decoder_update()
            encode_midi = False
            midi_buffer = None
            if self.state.last_frontend_payload is not None:
                button_values = self.state.last_frontend_payload.get("button_values") or {}
                encode_midi = bool(button_values.get("encodeMidi", False))
                midi_buffer = self.state.last_frontend_payload.get("midi_buffer")
            return self._build_response(midi_buffer=midi_buffer, encode_midi=encode_midi)

    def process_frame(self, payload: dict | None) -> dict:
        payload = payload or {}
        with self._lock:
            self.state.last_frontend_payload = payload
            slider_values = payload.get("slider_values") or {}
            button_values = payload.get("button_values") or {}

            cursor = np.asarray(payload.get("cursor", self.state.cursor), dtype=np.float32).reshape(2)
            points = np.asarray(payload.get("draggable_points", self.state.control_points), dtype=np.float32)
            if points.shape != self.state.control_points.shape:
                points = self.state.control_points

            grid = int(max(4, min(60, _safe_int(slider_values.get("gridResolution"), self.state.grid_resolution))))
            sigma = float(max(0.05, min(3.0, _safe_float(slider_values.get("sigma"), self.state.sigma))))
            heat_mode = str(button_values.get("heatmapMode", self.state.heatmap_mode))
            if heat_mode not in {"magnitude", "confidence"}:
                heat_mode = "magnitude"
            influence_mode = str(button_values.get("influenceMode", self.state.influence_mode))
            if influence_mode not in {"raw", "percentage"}:
                influence_mode = "raw"
            decode_enabled = bool(button_values.get("decodeEnabled", self.state.decode_enabled))

            self.state.cursor = cursor
            self.state.control_points = points
            self.state.grid_resolution = grid
            self.state.sigma = sigma
            self.state.heatmap_mode = heat_mode
            self.state.influence_mode = influence_mode
            self.state.decode_enabled = decode_enabled

            weights_signature = self._make_weights_signature(points, sigma)
            if weights_signature != self._last_weights_signature:
                self.weights = compute_rbf_weights(points, self.latent_vectors, sigma=sigma)
                self._last_weights_signature = weights_signature

            provided_latent = payload.get("latent_vector")
            use_provided_latent = bool(button_values.get("useProvidedLatent", False))
            if use_provided_latent and provided_latent is not None:
                latent_vector = np.asarray(provided_latent, dtype=np.float32).reshape(-1)
                expected_dim = int(self.latent_vectors.shape[1])
                if latent_vector.size != expected_dim:
                    raise ValueError(f"latent_vector must have length {expected_dim}, got {latent_vector.size}")
            else:
                latent_vector = rbf_interpolate_exact(points, self.weights, cursor, sigma=sigma).astype(np.float32)

            encode_midi = bool(button_values.get("encodeMidi", False))
            midi_buffer = payload.get("midi_buffer")
            if encode_midi and midi_buffer is not None:
                latent_vector = self.runtime.encode_midi_buffer_to_latent(midi_buffer)

            self.state.latent_vector = latent_vector.astype(np.float32)

            if self.state.decode_enabled:
                # Only restart the decoder when the latent actually changed.
                # In quantized mode the frontend only sends on cell change, so the signature
                # will always differ; in continuous mode this prevents redundant restarts.
                _decode_sig = (self.state.latent_vector.shape, np.round(self.state.latent_vector, 5).tobytes())
                if _decode_sig != self._last_triggered_latent_signature:
                    self.decoder_manager.request_decode(self.state.latent_vector, self.reconstruction_matrix)
                    self.decode_status = "decoding"
                    self._last_triggered_latent_signature = _decode_sig
                self._refresh_decoder_update()
            else:
                self.reconstruction_matrix.fill(0.0)
                self.decode_status = "idle"
                self.decode_progress_step = -1
                self._last_triggered_latent_signature = None

            heatmap_signature = self._make_heatmap_signature(points, sigma, grid, heat_mode)
            if heatmap_signature != self._last_heatmap_signature:
                self._cached_heatmap = self._compute_heatmap(points, self.weights, sigma, grid, heat_mode)
                self._last_heatmap_signature = heatmap_signature
            heatmap = self._cached_heatmap

            return self._build_response(midi_buffer=midi_buffer, encode_midi=encode_midi)


def create_app(sample_count: int = 10, sample_seed: int = 42):
    controller = RBFServerController(sample_count=sample_count, sample_seed=sample_seed)
    app = Flask(__name__, template_folder=os.path.dirname(__file__))
    socketio = SocketIO(app, cors_allowed_origins="*")
    assets_dir = (Path(__file__).resolve().parent.parent / "assets").resolve()

    def decode_broadcast_loop():
        while True:
            response = None
            with controller._lock:
                if controller.state.decode_enabled and controller._refresh_decoder_update():
                    response = controller._build_response(
                        midi_buffer=(controller.state.last_frontend_payload or {}).get("midi_buffer"),
                        encode_midi=bool(((controller.state.last_frontend_payload or {}).get("button_values") or {}).get("encodeMidi", False)),
                    )
            if response is not None:
                socketio.emit("frame", response)
            socketio.sleep(0.03)

    socketio.start_background_task(decode_broadcast_loop)

    @app.get("/")
    def index():
        return render_template("template.html")

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(str(assets_dir), filename)

    @app.get("/api/health")
    def health():
        return {"ok": True, "device": str(controller.runtime.device), "cuda_ready": torch.cuda.is_available()}

    @socketio.on("bootstrap")
    def on_bootstrap():
        payload = {
            "cursor": [0.0, 0.0],
            "draggable_points": controller.state.control_points.tolist(),
            "heatmap": None,
            "latent_vector": None,
            "midi_buffer": None,
            "influence_values": None,
            "button_values": {
                "heatmapMode": controller.state.heatmap_mode,
                "influenceMode": controller.state.influence_mode,
                "encodeMidi": False,
                "decodeEnabled": False,
            },
            "slider_values": {
                "gridResolution": controller.state.grid_resolution,
                "sigma": controller.state.sigma,
            },
        }
        response = controller.process_frame(payload)
        emit("bootstrap", response)

    @socketio.on("select_model")
    def on_select_model(payload):
        payload = payload or {}
        try:
            mode = str(payload.get("mode", controller.selected_mode)).lower()
            previous = int(payload.get("previous", 0))
            with controller._lock:
                controller.reload_model(mode=mode, previous=previous)
                controller._refresh_decoder_update()
                encode_midi = False
                midi_buffer = None
                if controller.state.last_frontend_payload is not None:
                    button_values = controller.state.last_frontend_payload.get("button_values") or {}
                    encode_midi = bool(button_values.get("encodeMidi", False))
                    midi_buffer = controller.state.last_frontend_payload.get("midi_buffer")
                response = controller._build_response(midi_buffer=midi_buffer, encode_midi=encode_midi)
            emit("bootstrap", response)
        except Exception as exc:
            emit("error", {"message": str(exc)})

    @socketio.on("frame")
    def on_frame(payload):
        try:
            response = controller.process_frame(payload)
            emit("frame", response)
        except Exception as exc:
            emit("error", {"message": str(exc)})

    return app, socketio


def main():
    AUTOMATIC_PORT_SELECTION = True
    if AUTOMATIC_PORT_SELECTION:
        # Automatic mode: CUDA machines default to 5006, others to 5009.
        if torch.cuda.is_available():
            port = 5006
        else:
            port = 5009
        user_input_string = str(port)
        user_input_int = port
        print(f"Automatic port selection enabled. CUDA available: {torch.cuda.is_available()}. Base port: {port}")
    else:
        user_input = input('Enter a port number between 1024 and 65535: ')
        user_input_string = user_input.strip()
        user_input_int = _safe_int(user_input_string, None)
        if user_input_int is not None and 1024 <= user_input_int <= 65535:
            port = user_input_int
        else:
            print(f"Invalid port number '{user_input_string}'. Falling back to default port 5006.")
            port = 5006
        print(f"user input string: '{user_input_string}', user input int: {user_input_int}, final port: {port}")



    host = os.environ.get("RBF_SERVER_HOST", "0.0.0.0")
    port = _safe_int(os.environ.get("RBF_SERVER_PORT", user_input_string), user_input_int)
    sample_count = _safe_int(os.environ.get("RBF_SAMPLE_COUNT", "10"), 10)
    sample_seed = _safe_int(os.environ.get("RBF_SAMPLE_SEED", "42"), 42)
    app, socketio = create_app(sample_count=sample_count, sample_seed=sample_seed)
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
