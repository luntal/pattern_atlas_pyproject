from __future__ import annotations

import os
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import re
from zoneinfo import ZoneInfo

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


def _clamp_sample_count(value: int, max_count: int) -> int:
    upper = max(1, int(max_count))
    return max(1, min(int(value), upper))


@dataclass
class FrontendState:
    cursor: np.ndarray
    control_points: np.ndarray
    heatmap_mode: str = "magnitude"
    influence_mode: str = "raw"
    decode_enabled: bool = False
    render_enabled: bool = False
    bpm_interp_mode: str = "gaussian_blend"
    latent_fallback_mode: str = "preserve_current_behavior"
    grid_resolution: int = 30
    sigma: float = 0.30
    latent_vector: np.ndarray | None = None
    current_bpm: float = 0.0
    last_frontend_payload: dict | None = None
    refined_blend_low: float = 0.0002
    refined_blend_high: float = 0.45
    sample_count: int = 10
    sample_randomize: bool = True
    sample_count_max: int = 40


_FALLBACK_BLEND_LOW = 0.001
_FALLBACK_BLEND_HIGH = 0.20
_FALLBACK_EPS = 1e-9
_ANCHOR_HIT_EPS = 1e-4
_LATENT_FALLBACK_MODES = (
    "preserve_current_behavior",
    "mean_latent",
    "nearest_anchor",
    "last_valid_latent",
    "user_defined_anchor0",
    "user_defined_anchor0_refined",
)
_ANCHOR0_HIDDEN_FALLBACK_MODES = {
    "user_defined_anchor0",
    "user_defined_anchor0_refined",
}

RETOKENIZE = False


def _uses_hidden_anchor0_mode(mode: str) -> bool:
    return str(mode) in _ANCHOR0_HIDDEN_FALLBACK_MODES


def _smoothstep01(t: float) -> float:
    t_clamped = max(0.0, min(1.0, float(t)))
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)


def _blend_factor_from_influence(influence_sum: float) -> float:
    if influence_sum <= _FALLBACK_BLEND_LOW:
        return 0.0
    if influence_sum >= _FALLBACK_BLEND_HIGH:
        return 1.0
    t = (float(influence_sum) - _FALLBACK_BLEND_LOW) / (_FALLBACK_BLEND_HIGH - _FALLBACK_BLEND_LOW)
    return _smoothstep01(t)


def _blend_factor_custom(influence_sum: float, low: float, high: float) -> float:
    if influence_sum <= low:
        return 0.0
    if influence_sum >= high:
        return 1.0
    t = (float(influence_sum) - float(low)) / (float(high) - float(low))
    return _smoothstep01(t)


def _lerp_vector(fallback_vec: np.ndarray, rbf_vec: np.ndarray, alpha: float) -> np.ndarray:
    a = float(max(0.0, min(1.0, alpha)))
    return ((1.0 - a) * fallback_vec + a * rbf_vec).astype(np.float32)


def _lerp_scalar(fallback_value: float, rbf_value: float, alpha: float) -> float:
    a = float(max(0.0, min(1.0, alpha)))
    return float((1.0 - a) * float(fallback_value) + a * float(rbf_value))


class FullFieldRenderManager:
    def __init__(self, runtime: "CudaRBFRuntime", hparams, retokenize: bool = False):
        self.runtime = runtime
        self.hparams = hparams
        self.retokenize = bool(retokenize)
        self._lock = threading.RLock()
        self._current_job_id = 0
        self._cancel_event = None
        self._latest_generation = 0
        self._last_consumed_generation = -1

        self._status = "idle"
        self._completed = 0
        self._total = 0
        self._output_path = None
        self._error = None

    def is_rendering(self) -> bool:
        with self._lock:
            return self._status == "rendering"

    def request_render(
        self,
        control_points: np.ndarray,
        anchor_bpms: np.ndarray,
        latent_weights: np.ndarray,
        bpm_weights: np.ndarray,
        bpm_interp_mode: str,
        latent_fallback_mode: str,
        fallback_mean_latent: np.ndarray | None,
        latent_vectors_snapshot: np.ndarray,
        weights_ex0: np.ndarray | None,
        bpm_weights_ex0: np.ndarray | None,
        anchor0_latent: np.ndarray | None,
        anchor0_bpm: float,
        sigma: float,
        grid_resolution: int,
        field_half_extent: float,
        branch_name: str,
        refined_blend_low: float = 0.0002,
        refined_blend_high: float = 0.45,
    ):
        with self._lock:
            self._current_job_id += 1
            job_id = self._current_job_id
            if self._cancel_event is not None:
                self._cancel_event.set()
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            total = int(grid_resolution) * int(grid_resolution)
            self._status = "rendering"
            self._completed = 0
            self._total = total
            self._output_path = None
            self._error = None
            self._latest_generation += 1

        thread = threading.Thread(
            target=self._render_job,
            args=(
                job_id,
                control_points.astype(np.float32, copy=True),
                anchor_bpms.astype(np.float32, copy=True),
                latent_weights.astype(np.float32, copy=True),
                bpm_weights.astype(np.float32, copy=True),
                str(bpm_interp_mode),
                str(latent_fallback_mode),
                fallback_mean_latent.astype(np.float32, copy=True) if fallback_mean_latent is not None else None,
                latent_vectors_snapshot.astype(np.float32, copy=True),
                weights_ex0.astype(np.float32, copy=True) if weights_ex0 is not None else None,
                bpm_weights_ex0.astype(np.float32, copy=True) if bpm_weights_ex0 is not None else None,
                anchor0_latent.astype(np.float32, copy=True) if anchor0_latent is not None else None,
                float(anchor0_bpm),
                float(sigma),
                int(grid_resolution),
                float(field_half_extent),
                safe_branch_name(branch_name),
                float(refined_blend_low),
                float(refined_blend_high),
                cancel_event,
            ),
            daemon=True,
        )
        thread.start()

    def cancel_render(self):
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()

    def consume_latest_update(self):
        with self._lock:
            if self._latest_generation == self._last_consumed_generation:
                return None
            self._last_consumed_generation = self._latest_generation
            return {
                "status": self._status,
                "completed": int(self._completed),
                "total": int(self._total),
                "output_path": self._output_path,
                "error": self._error,
            }

    def _publish_update(
        self,
        job_id: int,
        status: str,
        completed: int,
        total: int,
        output_path: str | None = None,
        error: str | None = None,
    ):
        with self._lock:
            if job_id != self._current_job_id:
                return
            self._status = status
            self._completed = int(completed)
            self._total = int(total)
            self._output_path = output_path
            self._error = error
            self._latest_generation += 1

    def _normalize_tokens(self, tokens: np.ndarray, target_len: int, pad_token: int) -> np.ndarray:
        out = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if out.size > target_len:
            out = out[:target_len]
        elif out.size < target_len:
            pad = np.full((target_len - out.size,), int(pad_token), dtype=np.int64)
            out = np.concatenate([out, pad], axis=0)
        return out

    def _render_job(
        self,
        job_id: int,
        control_points: np.ndarray,
        anchor_bpms: np.ndarray,
        latent_weights: np.ndarray,
        bpm_weights: np.ndarray,
        bpm_interp_mode: str,
        latent_fallback_mode: str,
        fallback_mean_latent: np.ndarray | None,
        latent_vectors_snapshot: np.ndarray,
        weights_ex0: np.ndarray | None,
        bpm_weights_ex0: np.ndarray | None,
        anchor0_latent: np.ndarray | None,
        anchor0_bpm: float,
        sigma: float,
        grid_resolution: int,
        field_half_extent: float,
        branch_name: str,
        refined_blend_low: float,
        refined_blend_high: float,
        cancel_event: threading.Event,
    ):
        total = int(grid_resolution) * int(grid_resolution)
        sequence_len = 256
        pad_token = int(self.hparams.pad_token)
        sequences_flat = []
        bpms = []
        completed = 0
        last_valid_latent: np.ndarray | None = None

        cell_size = (2.0 * field_half_extent) / max(1, int(grid_resolution))

        # Determine which control-point set and latent data to use per fallback mode.
        use_ex0 = (_uses_hidden_anchor0_mode(latent_fallback_mode)
                   and weights_ex0 is not None
                   and len(control_points) >= 2)
        interp_points = control_points[1:] if use_ex0 else control_points
        interp_weights = weights_ex0 if use_ex0 else latent_weights
        interp_bpm_weights = bpm_weights_ex0 if use_ex0 else bpm_weights
        interp_anchor_bpms = anchor_bpms[1:] if use_ex0 else anchor_bpms
        interp_latent_vectors = latent_vectors_snapshot[1:] if use_ex0 else latent_vectors_snapshot

        def _cell_latent(cursor: np.ndarray) -> np.ndarray:
            nonlocal last_valid_latent
            if latent_fallback_mode == "preserve_current_behavior":
                return rbf_interpolate_exact(control_points, latent_weights, cursor, sigma=sigma).astype(np.float32)

            influence = np.exp(
                -(np.linalg.norm(interp_points - cursor.reshape(1, 2), axis=1) ** 2) / (2 * sigma ** 2)
            ).astype(np.float32)
            influence_sum = float(np.sum(influence))
            rbf_vec = rbf_interpolate_exact(interp_points, interp_weights, cursor, sigma=sigma).astype(np.float32)

            if latent_fallback_mode == "last_valid_latent" and influence_sum >= _FALLBACK_BLEND_HIGH:
                last_valid_latent = rbf_vec.copy()

            if latent_fallback_mode == "user_defined_anchor0":
                fallback_vec = anchor0_latent if anchor0_latent is not None else fallback_mean_latent
            elif latent_fallback_mode == "user_defined_anchor0_refined":
                if influence.size == 0:
                    fallback_vec = anchor0_latent if anchor0_latent is not None else fallback_mean_latent
                else:
                    nearest_idx = int(np.argmin(np.linalg.norm(interp_points - cursor.reshape(1, 2), axis=1)))
                    nearest_dist = float(np.linalg.norm(interp_points[nearest_idx] - cursor))
                    if nearest_dist <= _ANCHOR_HIT_EPS:
                        target_vec = interp_latent_vectors[nearest_idx].astype(np.float32)
                    else:
                        denom = max(influence_sum, _FALLBACK_EPS)
                        target_vec = (influence.reshape(-1, 1) * interp_latent_vectors).sum(axis=0) / denom
                        target_vec = target_vec.astype(np.float32)
                    base_vec = anchor0_latent if anchor0_latent is not None else fallback_mean_latent
                    if base_vec is None:
                        return target_vec
                    alpha = _blend_factor_custom(influence_sum, refined_blend_low, refined_blend_high)
                    return _lerp_vector(base_vec, target_vec, alpha)
            elif latent_fallback_mode == "mean_latent":
                fallback_vec = fallback_mean_latent
            elif latent_fallback_mode == "nearest_anchor":
                denom = max(influence_sum, _FALLBACK_EPS)
                fallback_vec = (influence.reshape(-1, 1) * interp_latent_vectors).sum(axis=0) / denom
                fallback_vec = fallback_vec.astype(np.float32)
            elif latent_fallback_mode == "last_valid_latent":
                fallback_vec = last_valid_latent if last_valid_latent is not None else fallback_mean_latent
            else:
                return rbf_vec

            if fallback_vec is None:
                return rbf_vec
            alpha = _blend_factor_from_influence(influence_sum)
            return _lerp_vector(fallback_vec, rbf_vec, alpha)

        def _cell_bpm(cursor: np.ndarray) -> float:
            if use_ex0:
                infl = np.exp(
                    -(np.linalg.norm(interp_points - cursor.reshape(1, 2), axis=1) ** 2) / (2 * sigma ** 2)
                ).astype(np.float32)
                influence_sum = float(np.sum(infl))
                if latent_fallback_mode == "user_defined_anchor0_refined":
                    if infl.size == 0:
                        target_bpm = float(anchor0_bpm)
                    else:
                        nearest_idx = int(np.argmin(np.linalg.norm(interp_points - cursor.reshape(1, 2), axis=1)))
                        nearest_dist = float(np.linalg.norm(interp_points[nearest_idx] - cursor))
                        if nearest_dist <= _ANCHOR_HIT_EPS:
                            target_bpm = float(interp_anchor_bpms[nearest_idx])
                        else:
                            denom = max(influence_sum, _FALLBACK_EPS)
                            target_bpm = float(np.sum(infl * interp_anchor_bpms) / denom)
                    alpha = _blend_factor_custom(influence_sum, refined_blend_low, refined_blend_high)
                    return _lerp_scalar(float(anchor0_bpm), target_bpm, alpha)

                if bpm_interp_mode == "gaussian_blend":
                    denom = max(influence_sum, _FALLBACK_EPS)
                    rbf_bpm = float(np.sum(infl * interp_anchor_bpms) / denom)
                elif interp_bpm_weights is not None:
                    bpm = rbf_interpolate_exact(interp_points, interp_bpm_weights, cursor, sigma=sigma)
                    rbf_bpm = float(np.asarray(bpm).reshape(-1)[0])
                else:
                    rbf_bpm = float(interp_anchor_bpms[0]) if interp_anchor_bpms.size > 0 else 120.0
                alpha = _blend_factor_from_influence(influence_sum)
                return _lerp_scalar(float(anchor0_bpm), rbf_bpm, alpha)

            if bpm_interp_mode == "gaussian_blend":
                dists = np.linalg.norm(control_points - cursor.reshape(1, 2), axis=1)
                influence = np.exp(-(dists ** 2) / (2 * sigma ** 2)).astype(np.float32)
                denom = float(np.sum(influence))
                if denom > 0.0:
                    return float(np.sum(influence * anchor_bpms) / denom)
                return float(anchor_bpms[0]) if anchor_bpms.size > 0 else 120.0
            bpm = rbf_interpolate_exact(control_points, bpm_weights, cursor, sigma=sigma)
            return float(np.asarray(bpm).reshape(-1)[0])

        try:
            for row in range(int(grid_resolution)):
                wy = -field_half_extent + (row + 0.5) * cell_size
                for col in range(int(grid_resolution)):
                    if cancel_event.is_set():
                        self._publish_update(job_id, "cancelled", completed, total)
                        return

                    wx = -field_half_extent + (col + 0.5) * cell_size
                    cursor = np.asarray([wx, wy], dtype=np.float32)

                    latent = _cell_latent(cursor)
                    matrix, original_tokens = self.runtime.decode_latent_to_matrix(latent, return_tokens=True)
                    if self.retokenize:
                        tokens = self.runtime._tokens_from_matrix_buffer(matrix)
                    else:
                        tokens = original_tokens
                    normalized = self._normalize_tokens(tokens, sequence_len, pad_token)
                    sequences_flat.extend(normalized.tolist())
                    bpms.append(_cell_bpm(cursor))

                    completed += 1
                    self._publish_update(job_id, "rendering", completed, total)

            timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path("outputs") / branch_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"field_render_{timestamp}.npz"

            np.savez_compressed(
                str(output_path),
                sequences_1d=np.asarray(sequences_flat, dtype=np.int64),
                bpms_1d=np.asarray(bpms, dtype=np.float32),
                control_points=control_points.astype(np.float32),
                anchor_bpms=anchor_bpms.astype(np.float32),
                sequence_length=int(sequence_len),
                grid_resolution=int(grid_resolution),
                traversal_order="row_major",
                bpm_interp_mode=str(bpm_interp_mode),
                latent_fallback_mode=str(latent_fallback_mode),
                sigma=float(sigma),
                field_half_extent=float(field_half_extent),
                refined_blend_low=float(refined_blend_low),
                refined_blend_high=float(refined_blend_high),
                total_points=int(total),
                # Token vocabulary / layout metadata required for external token->MIDI decoding.
                step_token=int(self.hparams.step_token),
                eos_token=int(self.hparams.eos_token),
                pad_token=int(self.hparams.pad_token),
                sos_token=int(self.hparams.sos_token),
                min_pitch_token=int(self.hparams.min_pitch_token),
                max_pitch_token=int(self.hparams.max_pitch_token),
                min_vel_token=int(self.hparams.min_vel_token),
                max_vel_token=int(self.hparams.max_vel_token),
                min_timing_token=int(self.hparams.min_timing_token),
                max_timing_token=int(self.hparams.max_timing_token),
                center_timing_token=int(self.hparams.center_timing_token),
                n_vel=int(self.hparams.n_vel),
                n_timing=int(self.hparams.n_timing),
                width=int(self.hparams.width),
                height=int(self.hparams.height),
                max_seq_len=int(self.hparams.max_seq_len),
                dataset_name=str(getattr(self.hparams, "dataset_name", "")),
                branch=branch_name,
                timestamp=timestamp,
            )

            self._publish_update(job_id, "done", completed, total, output_path=str(output_path))
        except Exception as exc:
            self._publish_update(job_id, "error", completed, total, error=str(exc))


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
    def __init__(self, previous: int = 0, checkpoint_kind: str | None = None, retokenize: bool = False):
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
        self.retokenize = bool(retokenize)

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

    def decode_latent_to_matrix(self, latent_vector: np.ndarray, return_tokens: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
        generated_tokens: list[int] = []

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
                generated_tokens.append(token)

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

        if return_tokens:
            return matrix, np.asarray(generated_tokens, dtype=np.int64)
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

    def _data_branch_dir(self) -> Path:
        branch_name = safe_branch_name(get_current_branch())
        return Path("data") / branch_name

    def _list_available_datasets(self) -> list[str]:
        data_dir = self._data_branch_dir()
        if not data_dir.exists():
            return []
        return sorted([p.name for p in data_dir.glob("*.npz") if p.is_file()])

    def _resolve_dataset_name(self, preferred: str | None) -> str:
        options = self._list_available_datasets()
        if not options:
            raise FileNotFoundError(f"No .npz datasets found in {self._data_branch_dir()}")

        preferred_name = str(preferred) if preferred is not None else ""
        if preferred_name in options:
            return preferred_name

        hparams_default = str(getattr(self.hparams, "dataset_name", ""))
        if hparams_default in options:
            return hparams_default

        return options[0]

    def _compute_bpm_weights(self, control_points: np.ndarray, sigma: float) -> np.ndarray:
        bpm_targets = self.anchor_bpms.reshape(-1, 1).astype(np.float32)
        try:
            # Use zero ridge first so anchors reproduce their source BPM exactly.
            return compute_rbf_weights(control_points, bpm_targets, sigma=sigma, lambd=0.0)
        except np.linalg.LinAlgError:
            # Fall back to tiny ridge regularization for numerical stability.
            return compute_rbf_weights(control_points, bpm_targets, sigma=sigma, lambd=1e-6)

    def _recompute_fallback_caches(self):
        """Recompute caches used by fallback modes. Call after loading a new latent pool."""
        if len(self.latent_vectors) > 0:
            self._fallback_mean_latent: np.ndarray = np.mean(self.latent_vectors, axis=0).astype(np.float32)
        else:
            self._fallback_mean_latent = np.zeros(1, dtype=np.float32)

    def _compute_latent_with_fallback(
        self,
        cursor: np.ndarray,
        points: np.ndarray,
        weights: np.ndarray,
        sigma: float,
        mode: str,
    ) -> np.ndarray:
        """Interpolate a latent vector with selected fallback behaviour for low-influence regions."""
        if mode == "preserve_current_behavior":
            return rbf_interpolate_exact(points, weights, cursor, sigma=sigma).astype(np.float32)

        if mode == "user_defined_anchor0":
            # Exclude anchor 0 from interpolation; blend smoothly toward anchor 0 latent.
            anchor0_latent = self.latent_vectors[0].astype(np.float32)
            if len(points) < 2 or self.weights_ex0 is None:
                return anchor0_latent
            sub_points = points[1:]
            influence = self._compute_influence_raw(sub_points, cursor, sigma)
            influence_sum = float(np.sum(influence))
            rbf_vec = rbf_interpolate_exact(sub_points, self.weights_ex0, cursor, sigma=sigma).astype(np.float32)
            alpha = _blend_factor_from_influence(influence_sum)
            return _lerp_vector(anchor0_latent, rbf_vec, alpha)

        if mode == "user_defined_anchor0_refined":
            # Exclude anchor 0 and blend directly between anchor-0 latent and
            # normalized visible-anchor latent contribution.
            anchor0_latent = self.latent_vectors[0].astype(np.float32)
            if len(points) < 2:
                return anchor0_latent
            sub_points = points[1:]
            sub_latents = self.latent_vectors[1:]
            influence = self._compute_influence_raw(sub_points, cursor, sigma)
            influence_sum = float(np.sum(influence))
            if influence.size > 0:
                nearest_idx = int(np.argmin(np.linalg.norm(sub_points - cursor.reshape(1, 2), axis=1)))
                nearest_dist = float(np.linalg.norm(sub_points[nearest_idx] - cursor))
                if nearest_dist <= _ANCHOR_HIT_EPS:
                    return sub_latents[nearest_idx].astype(np.float32)

            denom = max(influence_sum, _FALLBACK_EPS)
            visible_target = (influence.reshape(-1, 1) * sub_latents).sum(axis=0) / denom
            visible_target = visible_target.astype(np.float32)
            alpha = _blend_factor_custom(influence_sum, self.state.refined_blend_low, self.state.refined_blend_high)
            return _lerp_vector(anchor0_latent, visible_target, alpha)

        # Remaining modes operate over the full control-point set.
        influence = self._compute_influence_raw(points, cursor, sigma)
        influence_sum = float(np.sum(influence))
        rbf_vec = rbf_interpolate_exact(points, weights, cursor, sigma=sigma).astype(np.float32)

        # Keep last-valid cache in pure RBF zone only.
        if mode == "last_valid_latent" and influence_sum >= _FALLBACK_BLEND_HIGH:
            self._fallback_last_valid_latent = rbf_vec.copy()

        fallback_vec: np.ndarray
        if mode == "mean_latent":
            fallback_vec = self._fallback_mean_latent.astype(np.float32)
        elif mode == "nearest_anchor":
            # Soft-nearest fallback: smooth Gaussian-weighted blend across anchors.
            denom = max(float(np.sum(influence)), _FALLBACK_EPS)
            fallback_vec = (influence.reshape(-1, 1) * self.latent_vectors).sum(axis=0) / denom
            fallback_vec = fallback_vec.astype(np.float32)
        elif mode == "last_valid_latent":
            if self._fallback_last_valid_latent is not None:
                fallback_vec = self._fallback_last_valid_latent.astype(np.float32)
            else:
                fallback_vec = self._fallback_mean_latent.astype(np.float32)
        else:
            return rbf_vec

        alpha = _blend_factor_from_influence(influence_sum)
        return _lerp_vector(fallback_vec, rbf_vec, alpha)

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

    def __init__(self, sample_count: int = 10, sample_seed: int = 42, sample_randomize: bool = True):
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.sample_randomize = bool(sample_randomize)
        self.sample_count_max = 40
        self.selected_mode = "none"
        self.selected_previous = 0

        for candidate in ["none", "best_val", "best_train", "last"]:
            if self._mode_files_static(candidate):
                self.selected_mode = candidate
                break

        checkpoint_kind = None if self.selected_mode == "none" else self.selected_mode

        self.runtime = CudaRBFRuntime(previous=self.selected_previous, checkpoint_kind=checkpoint_kind, retokenize=RETOKENIZE)
        self.hparams = self.runtime.hparams
        self._lock = threading.Lock()
        self.decoder_manager = RealtimeDecoderManager(self.runtime.decoder, self.hparams)
        self.render_manager = FullFieldRenderManager(self.runtime, self.hparams, retokenize=RETOKENIZE)
        self.dataset_name = self._resolve_dataset_name(getattr(self.hparams, "dataset_name", None))
        self.hparams.dataset_name = self.dataset_name

        self.model_catalog = self._build_model_catalog()
        self._normalize_selection_to_catalog()

        self._load_latent_pool()
        control_points = self._make_initial_control_points(len(self.latent_vectors), field_half_extent=2.0)
        self.field_half_extent = 2.0
        self.weights = compute_rbf_weights(control_points, self.latent_vectors, sigma=0.30)
        self.bpm_weights = self._compute_bpm_weights(control_points, sigma=0.30)
        self.weights_ex0: np.ndarray | None = None
        self.bpm_weights_ex0: np.ndarray | None = None
        self._fallback_last_valid_latent: np.ndarray | None = None
        self._last_weights_signature = None
        self._last_heatmap_signature = None
        self._cached_heatmap = np.zeros((30, 30), dtype=np.float32)
        self.recon_cols = int(self.hparams.width * self.hparams.n_timing)
        self.reconstruction_matrix = np.zeros((int(self.hparams.height), self.recon_cols), dtype=np.float32)
        self.decode_status = "idle"
        self.decode_progress_step = -1
        self.render_status = "idle"
        self.render_completed = 0
        self.render_total = 0
        self.render_output_path = None
        self.render_error = None
        self._last_triggered_latent_signature = None
        self.state = FrontendState(
            cursor=np.zeros(2, dtype=np.float32),
            control_points=control_points,
            heatmap_mode="magnitude",
            influence_mode="raw",
            bpm_interp_mode="gaussian_blend",
            render_enabled=False,
            grid_resolution=30,
            sigma=0.30,
            current_bpm=float(self.anchor_bpms[0]) if len(self.anchor_bpms) > 0 else 0.0,
            sample_count=self.sample_count,
            sample_randomize=self.sample_randomize,
            sample_count_max=self.sample_count_max,
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
        # Keep dataset order intact here; optional randomization is controlled below.
        self.hparams.dataset_name = self.dataset_name
        data, metadata = load_dataset(self.hparams, shuffle=False)
        dataset_size = int(len(data))
        self.sample_count_max = max(1, min(40, dataset_size))
        self.sample_count = _clamp_sample_count(self.sample_count, self.sample_count_max)
        data, metadata, _ = pick_reproducible_samples(
            data,
            metadata,
            sample_count=self.sample_count,
            sample_seed=self.sample_seed,
            randomize=self.sample_randomize,
        )
        tokens = torch.from_numpy(data).to(self.runtime.device, dtype=torch.long)
        beat_tokens = torch.cumsum((tokens == int(self.hparams.step_token)).to(torch.long), dim=1)
        beat_tokens += 1
        beat_tokens[:, 0] = 0
        beat_tokens = beat_tokens.masked_fill(tokens == int(self.hparams.pad_token), 0)
        with torch.no_grad():
            mu, _ = self.runtime.encoder(tokens, beat_tokens)
        self.latent_vectors = mu.detach().cpu().numpy().astype(np.float32)

        bpms = []
        labels = []
        for item in metadata:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
                filename = os.path.basename(str(item[0]))
                filename_no_ext = os.path.splitext(filename)[0]
                bpm = item[2] if len(item) > 2 else None
                bpm_value = _safe_float(bpm, -1.0)
                if bpm_value > 0:
                    labels.append(f"{filename_no_ext} (BPM: {bpm_value:.2f})")
                    bpms.append(float(bpm_value))
                else:
                    labels.append(filename_no_ext)
                    bpms.append(120.0)
            else:
                labels.append(str(item))
                bpms.append(120.0)
        self.labels = labels
        self.anchor_bpms = np.asarray(bpms, dtype=np.float32)
        self._recompute_fallback_caches()

    def _rebuild_pool_dependent_state(self, reset_modes: bool = False):
        self.render_manager.cancel_render()
        control_points = self._make_initial_control_points(len(self.latent_vectors), field_half_extent=self.field_half_extent)
        self.state.control_points = control_points
        self.state.cursor = np.zeros(2, dtype=np.float32)
        self.state.latent_vector = None
        self.state.current_bpm = float(self.anchor_bpms[0]) if len(self.anchor_bpms) > 0 else 0.0
        self.state.decode_enabled = False
        self.state.render_enabled = False
        self.state.sample_count = int(self.sample_count)
        self.state.sample_randomize = bool(self.sample_randomize)
        self.state.sample_count_max = int(self.sample_count_max)
        if reset_modes:
            self.state.bpm_interp_mode = "gaussian_blend"
            self.state.latent_fallback_mode = "preserve_current_behavior"

        self.recon_cols = int(self.hparams.width * self.hparams.n_timing)
        self.reconstruction_matrix = np.zeros((int(self.hparams.height), self.recon_cols), dtype=np.float32)
        self.decode_status = "idle"
        self.decode_progress_step = -1
        self.render_status = "idle"
        self.render_completed = 0
        self.render_total = 0
        self.render_output_path = None
        self.render_error = None
        self._last_weights_signature = None
        self._last_heatmap_signature = None
        self._last_triggered_latent_signature = None
        self.weights_ex0 = None
        self.bpm_weights_ex0 = None
        self._fallback_last_valid_latent = None
        self.weights = compute_rbf_weights(control_points, self.latent_vectors, sigma=self.state.sigma)
        self.bpm_weights = self._compute_bpm_weights(control_points, sigma=self.state.sigma)
        self._cached_heatmap = np.zeros((self.state.grid_resolution, self.state.grid_resolution), dtype=np.float32)

    def reload_model(self, mode: str, previous: int = 0):
        if mode not in self.MODE_LABELS:
            raise ValueError(f"Unknown mode '{mode}'.")

        previous_points = self.state.control_points.astype(np.float32, copy=True)
        previous_cursor = self.state.cursor.astype(np.float32, copy=True)
        previous_latent_fallback_mode = str(self.state.latent_fallback_mode)
        previous_dataset_name = str(self.dataset_name)

        files = self._mode_files(mode)
        if not files:
            raise ValueError(f"No checkpoints available for mode '{mode}'.")

        previous = max(0, int(previous))
        if previous >= len(files):
            previous = len(files) - 1

        checkpoint_kind = None if mode == "none" else mode
        self.runtime = CudaRBFRuntime(previous=previous, checkpoint_kind=checkpoint_kind, retokenize=RETOKENIZE)
        self.hparams = self.runtime.hparams
        self.decoder_manager = RealtimeDecoderManager(self.runtime.decoder, self.hparams)
        self.render_manager = FullFieldRenderManager(self.runtime, self.hparams, retokenize=RETOKENIZE)
        self.dataset_name = self._resolve_dataset_name(previous_dataset_name)
        self.hparams.dataset_name = self.dataset_name
        self._load_latent_pool()

        self.selected_mode = mode
        self.selected_previous = previous
        self.model_catalog = self._build_model_catalog()
        self._normalize_selection_to_catalog()
        self._rebuild_pool_dependent_state(reset_modes=True)

        # Keep the user's currently selected latent fallback mode when switching models.
        if previous_latent_fallback_mode in _LATENT_FALLBACK_MODES:
            self.state.latent_fallback_mode = previous_latent_fallback_mode

        # Preserve user-adjusted anchor positions across checkpoint switches.
        if previous_points.ndim == 2 and previous_points.shape[1] == 2 and len(self.latent_vectors) > 0:
            n_new = int(len(self.latent_vectors))
            preserved_points = self._make_initial_control_points(n_new, field_half_extent=self.field_half_extent)
            keep = min(int(previous_points.shape[0]), n_new)
            if keep > 0:
                preserved_points[:keep] = previous_points[:keep]

            self.state.control_points = preserved_points.astype(np.float32)
            if previous_cursor.shape == (2,):
                self.state.cursor = previous_cursor.astype(np.float32)

            self.weights = compute_rbf_weights(self.state.control_points, self.latent_vectors, sigma=self.state.sigma)
            self.bpm_weights = self._compute_bpm_weights(self.state.control_points, sigma=self.state.sigma)

            if len(self.state.control_points) >= 2:
                sub_pts = self.state.control_points[1:]
                try:
                    self.weights_ex0 = compute_rbf_weights(sub_pts, self.latent_vectors[1:], sigma=self.state.sigma)
                except np.linalg.LinAlgError:
                    self.weights_ex0 = None
                sub_bpms = self.anchor_bpms[1:].reshape(-1, 1).astype(np.float32)
                try:
                    self.bpm_weights_ex0 = compute_rbf_weights(sub_pts, sub_bpms, sigma=self.state.sigma, lambd=0.0)
                except np.linalg.LinAlgError:
                    try:
                        self.bpm_weights_ex0 = compute_rbf_weights(sub_pts, sub_bpms, sigma=self.state.sigma, lambd=1e-6)
                    except np.linalg.LinAlgError:
                        self.bpm_weights_ex0 = None
            else:
                self.weights_ex0 = None
                self.bpm_weights_ex0 = None

            if _uses_hidden_anchor0_mode(self.state.latent_fallback_mode) and len(self.state.control_points) > 1 and self.weights_ex0 is not None:
                heatmap_points = self.state.control_points[1:]
                heatmap_weights = self.weights_ex0
            else:
                heatmap_points = self.state.control_points
                heatmap_weights = self.weights

            self._cached_heatmap = self._compute_heatmap(
                heatmap_points,
                heatmap_weights,
                self.state.sigma,
                self.state.grid_resolution,
                self.state.heatmap_mode,
            )
            self._last_weights_signature = self._make_weights_signature(self.state.control_points, self.state.sigma)
            self._last_heatmap_signature = self._make_heatmap_signature(
                heatmap_points,
                self.state.sigma,
                self.state.grid_resolution,
                self.state.heatmap_mode,
                self.state.latent_fallback_mode,
            )

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

    def _interpolate_bpm(
        self,
        control_points: np.ndarray,
        anchor_bpms: np.ndarray,
        bpm_weights: np.ndarray,
        cursor: np.ndarray,
        sigma: float,
        mode: str,
        fallback_mode: str = "preserve_current_behavior",
    ) -> float:
        # For hidden-anchor-0 modes, interpolate over anchors [1:] and blend to anchor 0 BPM.
        if _uses_hidden_anchor0_mode(fallback_mode) and len(control_points) >= 2:
            sub_points = control_points[1:]
            sub_bpms = anchor_bpms[1:]
            sub_influence = self._compute_influence_raw(sub_points, cursor, sigma)
            influence_sum = float(np.sum(sub_influence))
            if fallback_mode == "user_defined_anchor0_refined":
                if sub_influence.size > 0:
                    nearest_idx = int(np.argmin(np.linalg.norm(sub_points - cursor.reshape(1, 2), axis=1)))
                    nearest_dist = float(np.linalg.norm(sub_points[nearest_idx] - cursor))
                    if nearest_dist <= _ANCHOR_HIT_EPS:
                        target_bpm = float(sub_bpms[nearest_idx])
                    else:
                        denom = max(influence_sum, _FALLBACK_EPS)
                        target_bpm = float(np.sum(sub_influence * sub_bpms) / denom)
                else:
                    target_bpm = float(anchor_bpms[0]) if anchor_bpms.size > 0 else 120.0
                fallback_bpm = float(anchor_bpms[0]) if anchor_bpms.size > 0 else 120.0
                alpha = _blend_factor_custom(influence_sum, self.state.refined_blend_low, self.state.refined_blend_high)
                return _lerp_scalar(fallback_bpm, target_bpm, alpha)

            if mode == "gaussian_blend":
                denom = max(influence_sum, _FALLBACK_EPS)
                rbf_bpm = float(np.sum(sub_influence * sub_bpms) / denom)
            elif self.bpm_weights_ex0 is not None:
                bpm = rbf_interpolate_exact(sub_points, self.bpm_weights_ex0, cursor, sigma=sigma)
                rbf_bpm = float(np.asarray(bpm).reshape(-1)[0])
            else:
                rbf_bpm = float(sub_bpms[0]) if sub_bpms.size > 0 else 120.0

            fallback_bpm = float(anchor_bpms[0]) if anchor_bpms.size > 0 else 120.0
            alpha = _blend_factor_from_influence(influence_sum)
            return _lerp_scalar(fallback_bpm, rbf_bpm, alpha)

        if mode == "gaussian_blend":
            influence = self._compute_influence_raw(control_points, cursor, sigma)
            denom = float(np.sum(influence))
            if denom > 0.0:
                return float(np.sum(influence * anchor_bpms) / denom)
            if anchor_bpms.size > 0:
                return float(anchor_bpms[0])
            return 120.0

        bpm = rbf_interpolate_exact(control_points, bpm_weights, cursor, sigma=sigma)
        return float(np.asarray(bpm).reshape(-1)[0])

    def _make_weights_signature(self, points: np.ndarray, sigma: float):
        return (points.shape, np.round(points, 4).tobytes(), round(float(sigma), 4))

    def _make_heatmap_signature(self, points: np.ndarray, sigma: float, grid: int, mode: str, fallback_mode: str = ""):
        return (points.shape, np.round(points, 4).tobytes(), round(float(sigma), 4), int(grid), str(mode), str(fallback_mode))

    def _refresh_decoder_update(self) -> bool:
        decode_update = self.decoder_manager.consume_latest_update()
        if decode_update is None:
            return False
        self.reconstruction_matrix = decode_update["matrix"]
        self.decode_status = decode_update["status"]
        self.decode_progress_step = int(decode_update["progress_step"])
        return True

    def _refresh_render_update(self) -> bool:
        render_update = self.render_manager.consume_latest_update()
        if render_update is None:
            return False
        self.render_status = render_update["status"]
        self.render_completed = int(render_update["completed"])
        self.render_total = int(render_update["total"])
        self.render_output_path = render_update["output_path"]
        self.render_error = render_update["error"]
        if self.render_status in {"done", "cancelled", "error", "idle"}:
            self.state.render_enabled = False
        return True

    def _build_response(self, midi_buffer, encode_midi: bool) -> dict:
        points = self.state.control_points
        cursor = self.state.cursor
        sigma = self.state.sigma
        influence_mode = self.state.influence_mode
        hidden_anchor0_mode = _uses_hidden_anchor0_mode(self.state.latent_fallback_mode) and len(points) > 1

        # In hidden-anchor-0 modes, anchor 0 is internal-only fallback and hidden from UI.
        influence_points = points[1:] if hidden_anchor0_mode else points
        influence_raw = self._compute_influence_raw(influence_points, cursor, sigma)
        total = float(np.sum(influence_raw))
        influence_pct = influence_raw / total if total > 0.0 else np.zeros_like(influence_raw)
        response_points = points[1:] if hidden_anchor0_mode else points
        response_labels = self.labels[1:] if hidden_anchor0_mode else self.labels
        response_anchor_bpms = self.anchor_bpms[1:] if hidden_anchor0_mode else self.anchor_bpms

        return {
            "request_id": (self.state.last_frontend_payload or {}).get("client_request_id"),
            "cursor": cursor.tolist(),
            "draggable_points": response_points.tolist(),
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
                "bpmInterpMode": self.state.bpm_interp_mode,
                "latentFallbackMode": self.state.latent_fallback_mode,
                "datasetName": self.dataset_name,
                "encodeMidi": encode_midi,
                "decodeEnabled": self.state.decode_enabled,
                "renderEnabled": self.state.render_enabled,
                "sampleRandomize": self.state.sample_randomize,
            },
            "anchor0_locked": False,
            "slider_values": {
                "gridResolution": self.state.grid_resolution,
                "sigma": sigma,
                "refinedBlendLow": self.state.refined_blend_low,
                "refinedBlendHigh": self.state.refined_blend_high,
                "sampleCount": int(self.state.sample_count),
                "sampleCountMax": int(self.state.sample_count_max),
            },
            "labels": response_labels,
            "anchor_bpms": response_anchor_bpms.tolist(),
            "current_bpm": float(self.state.current_bpm),
            "cuda_ready": True,
            "field_half_extent": self.field_half_extent,
            "n_timing": int(self.hparams.n_timing),
            "recon_cols": int(self.hparams.width * self.hparams.n_timing),
            "height": int(self.hparams.height),
            "decode_status": self.decode_status,
            "decode_progress_step": int(self.decode_progress_step),
            "render_status": self.render_status,
            "render_completed": int(self.render_completed),
            "render_total": int(self.render_total),
            "render_output_path": self.render_output_path,
            "render_error": self.render_error,
            "model_options": self.model_catalog,
            "model_selection": {
                "mode": self.selected_mode,
                "previous": int(self.selected_previous),
                "timestamp": self._selected_timestamp(),
            },
            "dataset_options": self._list_available_datasets(),
            "dataset_selection": self.dataset_name,
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
            self._refresh_render_update()
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
            self._refresh_render_update()
            self.state.last_frontend_payload = payload
            slider_values = payload.get("slider_values") or {}
            button_values = payload.get("button_values") or {}

            cursor = np.asarray(payload.get("cursor", self.state.cursor), dtype=np.float32).reshape(2)
            raw_points = np.asarray(payload.get("draggable_points", self.state.control_points), dtype=np.float32)

            grid = int(max(4, min(60, _safe_int(slider_values.get("gridResolution"), self.state.grid_resolution))))
            sigma = float(max(0.05, min(3.0, _safe_float(slider_values.get("sigma"), self.state.sigma))))
            heat_mode = str(button_values.get("heatmapMode", self.state.heatmap_mode))
            if heat_mode not in {"magnitude", "confidence"}:
                heat_mode = "magnitude"
            influence_mode = str(button_values.get("influenceMode", self.state.influence_mode))
            if influence_mode not in {"raw", "percentage"}:
                influence_mode = "raw"
            bpm_interp_mode = str(button_values.get("bpmInterpMode", self.state.bpm_interp_mode))
            if bpm_interp_mode not in {"exact_rbf", "gaussian_blend"}:
                bpm_interp_mode = "gaussian_blend"
            latent_fallback_mode = str(button_values.get("latentFallbackMode", self.state.latent_fallback_mode))
            if latent_fallback_mode not in _LATENT_FALLBACK_MODES:
                latent_fallback_mode = "preserve_current_behavior"
            decode_enabled = bool(button_values.get("decodeEnabled", self.state.decode_enabled))
            render_enabled = bool(button_values.get("renderEnabled", self.state.render_enabled))
            # Guard against stale heartbeat re-arming the render when the job just finished.
            # _refresh_render_update() above may have just flipped render_status to a terminal
            # state (done/cancelled/error) in this same call, but the heartbeat payload still
            # carries renderEnabled=true because the frontend hasn't received the update yet.
            # Without this guard the False→True transition triggers request_render() again.
            if self.render_status in {"done", "cancelled", "error"}:
                render_enabled = False
            sample_count = _clamp_sample_count(
                _safe_int(slider_values.get("sampleCount"), self.state.sample_count),
                self.state.sample_count_max,
            )
            sample_randomize = bool(button_values.get("sampleRandomize", self.state.sample_randomize))
            dataset_name = str(button_values.get("datasetName", self.dataset_name))

            sample_pool_changed = False
            if sample_count != self.sample_count or sample_randomize != self.sample_randomize:
                sample_pool_changed = True
                self.sample_count = sample_count
                self.sample_randomize = sample_randomize
                self._load_latent_pool()
                self._rebuild_pool_dependent_state(reset_modes=False)
                # Ignore stale point/cursor data from this frame after pool reset.
                cursor = self.state.cursor.copy()
                raw_points = self.state.control_points.copy()
                decode_enabled = False
                render_enabled = False

            if dataset_name != self.dataset_name:
                self.dataset_name = self._resolve_dataset_name(dataset_name)
                self.hparams.dataset_name = self.dataset_name
                self._load_latent_pool()
                self._rebuild_pool_dependent_state(reset_modes=False)
                # Ignore stale point/cursor data from this frame after dataset swap.
                cursor = self.state.cursor.copy()
                raw_points = self.state.control_points.copy()
                decode_enabled = False
                render_enabled = False

            raw_blend_low = _safe_float(slider_values.get("refinedBlendLow"), self.state.refined_blend_low)
            raw_blend_high = _safe_float(slider_values.get("refinedBlendHigh"), self.state.refined_blend_high)
            refined_blend_low = float(max(1e-5, min(0.1, raw_blend_low)))
            refined_blend_high = float(max(0.05, min(1.5, raw_blend_high)))
            if refined_blend_high <= refined_blend_low:
                refined_blend_high = refined_blend_low + 0.01

            # Rebuild full internal control points when UI hides anchor 0 in user_defined mode.
            expected_shape = self.state.control_points.shape
            points = self.state.control_points
            if not sample_pool_changed and raw_points.ndim == 2 and raw_points.shape[1] == 2:
                if _uses_hidden_anchor0_mode(latent_fallback_mode):
                    full_n = int(expected_shape[0])
                    if raw_points.shape == expected_shape:
                        # Transition frame can still carry full-N points from previous mode.
                        points = raw_points
                    elif full_n > 0 and raw_points.shape[0] == full_n - 1:
                        # Hidden-anchor UI payload (N-1): restore full internal array by
                        # keeping anchor 0 from prior state and prepending it.
                        points = np.vstack([self.state.control_points[0:1], raw_points]).astype(np.float32)
                elif raw_points.shape == expected_shape:
                    points = raw_points

            prev_render_enabled = bool(self.state.render_enabled)

            self.state.cursor = cursor
            self.state.control_points = points
            self.state.grid_resolution = grid
            self.state.sigma = sigma
            self.state.heatmap_mode = heat_mode
            self.state.influence_mode = influence_mode
            self.state.bpm_interp_mode = bpm_interp_mode
            self.state.latent_fallback_mode = latent_fallback_mode
            self.state.decode_enabled = decode_enabled
            self.state.render_enabled = render_enabled
            self.state.refined_blend_low = refined_blend_low
            self.state.refined_blend_high = refined_blend_high
            self.state.sample_count = int(self.sample_count)
            self.state.sample_randomize = bool(self.sample_randomize)
            self.state.sample_count_max = int(self.sample_count_max)

            weights_signature = self._make_weights_signature(points, sigma)
            if weights_signature != self._last_weights_signature:
                self.weights = compute_rbf_weights(points, self.latent_vectors, sigma=sigma)
                self.bpm_weights = self._compute_bpm_weights(points, sigma=sigma)
                # Precompute ex0 weights for user_defined_anchor0 mode (anchors [1:]).
                if len(points) >= 2:
                    sub_pts = points[1:]
                    try:
                        self.weights_ex0 = compute_rbf_weights(sub_pts, self.latent_vectors[1:], sigma=sigma)
                    except np.linalg.LinAlgError:
                        self.weights_ex0 = None
                    sub_bpms = self.anchor_bpms[1:].reshape(-1, 1).astype(np.float32)
                    try:
                        self.bpm_weights_ex0 = compute_rbf_weights(sub_pts, sub_bpms, sigma=sigma, lambd=0.0)
                    except np.linalg.LinAlgError:
                        try:
                            self.bpm_weights_ex0 = compute_rbf_weights(sub_pts, sub_bpms, sigma=sigma, lambd=1e-6)
                        except np.linalg.LinAlgError:
                            self.bpm_weights_ex0 = None
                else:
                    self.weights_ex0 = None
                    self.bpm_weights_ex0 = None
                self._last_weights_signature = weights_signature

            provided_latent = payload.get("latent_vector")
            use_provided_latent = bool(button_values.get("useProvidedLatent", False))
            if use_provided_latent and provided_latent is not None:
                latent_vector = np.asarray(provided_latent, dtype=np.float32).reshape(-1)
                expected_dim = int(self.latent_vectors.shape[1])
                if latent_vector.size != expected_dim:
                    raise ValueError(f"latent_vector must have length {expected_dim}, got {latent_vector.size}")
            else:
                latent_vector = self._compute_latent_with_fallback(
                    cursor=cursor,
                    points=points,
                    weights=self.weights,
                    sigma=sigma,
                    mode=latent_fallback_mode,
                )

            self.state.current_bpm = self._interpolate_bpm(
                control_points=points,
                anchor_bpms=self.anchor_bpms,
                bpm_weights=self.bpm_weights,
                cursor=cursor,
                sigma=sigma,
                mode=self.state.bpm_interp_mode,
                fallback_mode=latent_fallback_mode,
            )

            if (not prev_render_enabled) and self.state.render_enabled:
                # Start a new background render with current interpolation settings.
                self.state.decode_enabled = False
                self.decode_status = "idle"
                self.decode_progress_step = -1
                self._last_triggered_latent_signature = None
                self.reconstruction_matrix.fill(0.0)
                self.render_manager.request_render(
                    control_points=points,
                    anchor_bpms=self.anchor_bpms,
                    latent_weights=self.weights,
                    bpm_weights=self.bpm_weights,
                    bpm_interp_mode=self.state.bpm_interp_mode,
                    latent_fallback_mode=latent_fallback_mode,
                    fallback_mean_latent=self._fallback_mean_latent.copy(),
                    latent_vectors_snapshot=self.latent_vectors.copy(),
                    weights_ex0=self.weights_ex0.copy() if self.weights_ex0 is not None else None,
                    bpm_weights_ex0=self.bpm_weights_ex0.copy() if self.bpm_weights_ex0 is not None else None,
                    anchor0_latent=self.latent_vectors[0].copy() if len(self.latent_vectors) > 0 else None,
                    anchor0_bpm=float(self.anchor_bpms[0]) if len(self.anchor_bpms) > 0 else 120.0,
                    sigma=sigma,
                    grid_resolution=grid,
                    field_half_extent=self.field_half_extent,
                    branch_name=get_current_branch(),
                    refined_blend_low=self.state.refined_blend_low,
                    refined_blend_high=self.state.refined_blend_high,
                )
            elif prev_render_enabled and (not self.state.render_enabled):
                # Cancel current render when toggle is switched off.
                # Do not gate by cached render_status: a quick toggle can land
                # between status updates and otherwise miss cancellation.
                self.render_manager.cancel_render()

            self._refresh_render_update()

            if self.render_status == "rendering":
                # Keep UI responsive by preventing decode work during full-field rendering.
                self.state.decode_enabled = False

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

            if _uses_hidden_anchor0_mode(latent_fallback_mode) and len(points) > 1 and self.weights_ex0 is not None:
                heatmap_points = points[1:]
                heatmap_weights = self.weights_ex0
            else:
                heatmap_points = points
                heatmap_weights = self.weights
            heatmap_signature = self._make_heatmap_signature(heatmap_points, sigma, grid, heat_mode, latent_fallback_mode)
            if heatmap_signature != self._last_heatmap_signature:
                self._cached_heatmap = self._compute_heatmap(heatmap_points, heatmap_weights, sigma, grid, heat_mode)
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

    def render_broadcast_loop():
        while True:
            response = None
            with controller._lock:
                if controller._refresh_render_update():
                    response = controller._build_response(
                        midi_buffer=(controller.state.last_frontend_payload or {}).get("midi_buffer"),
                        encode_midi=bool(((controller.state.last_frontend_payload or {}).get("button_values") or {}).get("encodeMidi", False)),
                    )
            if response is not None:
                socketio.emit("frame", response)
            socketio.sleep(0.03)

    socketio.start_background_task(decode_broadcast_loop)
    socketio.start_background_task(render_broadcast_loop)

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
                "bpmInterpMode": controller.state.bpm_interp_mode,
                "latentFallbackMode": controller.state.latent_fallback_mode,
                "datasetName": controller.dataset_name,
                "encodeMidi": False,
                "decodeEnabled": False,
                "renderEnabled": False,
                "sampleRandomize": controller.state.sample_randomize,
            },
            "slider_values": {
                "gridResolution": controller.state.grid_resolution,
                "sigma": controller.state.sigma,
                "sampleCount": controller.state.sample_count,
                "sampleCountMax": controller.state.sample_count_max,
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
    global RETOKENIZE
    RETOKENIZE = False
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
