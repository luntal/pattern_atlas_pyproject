from __future__ import annotations

import atexit
import contextlib
import gc
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import re
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

from hparams import hparams as _DEFAULT_HPARAMS
from modules.branch_paths import get_current_branch, project_root, safe_branch_name
from modules import compound, dataset_tools
from modules.distributed import cuda_device_list, device_roles, inference_device
from modules.hparams_io import HparamsSaveError, read_hparams, save_hparams
from modules.radial_basis_function import compute_rbf_weights, rbf_interpolate_exact
from modules.train_params import ALLOWED_KEYS, ARCHITECTURE_KEYS, MODEL_KEYS, TRAIN_KEYS, coerce_param
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


def _safe_rbf_weights(points: np.ndarray, targets: np.ndarray, sigma: float, **kwargs) -> np.ndarray:
    """compute_rbf_weights that tolerates an empty anchor set (no model / no dataset)."""
    targets = np.asarray(targets, dtype=np.float32)
    if len(points) == 0 or targets.shape[0] == 0:
        cols = int(targets.shape[1]) if targets.ndim > 1 else 1
        return np.zeros((0, cols), dtype=np.float32)
    return compute_rbf_weights(points, targets, sigma=sigma, **kwargs)


def _clamp_sample_count(value: int, max_count: int) -> int:
    upper = max(1, int(max_count))
    return max(1, min(int(value), upper))


def _clamp_point_to_extent(point: np.ndarray, half_extent: float) -> np.ndarray:
    h = float(max(0.0, half_extent))
    p = np.asarray(point, dtype=np.float32).reshape(2)
    return np.clip(p, -h, h).astype(np.float32)


def _clamp_points_to_extent(points: np.ndarray, half_extent: float) -> np.ndarray:
    h = float(max(0.0, half_extent))
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return pts
    return np.clip(pts, -h, h).astype(np.float32)


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
# Number of grid cells decoded together (batched greedy decoding with KV-cache) in the full-field render.
# Can be overridden via the RBF_DECODE_BATCH_SIZE environment variable.
DECODE_BATCH_SIZE = max(1, int(os.environ.get("RBF_DECODE_BATCH_SIZE", "64")))
# Real-time decoding (cursor movement): number of parallel decode workers. Running jobs are never cancelled by a newer
# request; at most one (the newest) request waits, and a result is only shown when no newer result is shown already.
DECODE_WORKERS = max(1, int(os.environ.get("RBF_DECODE_WORKERS", "3")))
# 1 = additionally publish partial results while a job runs (live note by note, but a newer job then supersedes older ones).
DECODE_STREAM = str(os.environ.get("RBF_DECODE_STREAM", "0")).strip().lower() in {"1", "true", "yes", "on"}
# The GPU -> CPU 'all finished' test of the generation loop only runs every N steps (saves syncs).
DECODE_CHECK_EVERY = max(1, int(os.environ.get("RBF_DECODE_CHECK_EVERY", "8")))
DEBUG_CONTROL_POINTS = str(os.environ.get("RBF_DEBUG_CONTROL_POINTS", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_print_control_points(tag: str, points: np.ndarray, extra: str = ""):
    if not DEBUG_CONTROL_POINTS:
        return
    pts = np.asarray(points, dtype=np.float32)
    pts_str = np.array2string(
        pts,
        precision=4,
        floatmode="fixed",
        suppress_small=False,
        max_line_width=200,
    )
    suffix = f" | {extra}" if extra else ""
    print(f"[control_points] {tag} | n={int(pts.shape[0])}{suffix} | positions={pts_str}", flush=True)


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


_PITCH_INFO_CACHE: dict = {}


def _pitch_info(n_pitch: int) -> dict:
    """Pitch labels and chromatic -> generic MIDI mapping for the reconstruction viewer (cached)."""
    n_pitch = int(n_pitch)
    if n_pitch not in _PITCH_INFO_CACHE:
        try:
            labels = [str(x) for x in dataset_tools.pitch_labels()][:n_pitch]
        except Exception:
            labels = []
        labels += [""] * (n_pitch - len(labels))
        try:
            from modules.midi_mappings import chromatic_to_generic
            generic = [int(chromatic_to_generic(i)) for i in range(n_pitch)]
        except Exception:
            generic = [i + 36 for i in range(n_pitch)]
        _PITCH_INFO_CACHE[n_pitch] = {"pitch_labels": labels, "generic_pitches": generic}
    return _PITCH_INFO_CACHE[n_pitch]


class FullFieldRenderManager:
    def __init__(self, runtime: "CudaRBFRuntime", hparams, retokenize: bool = False, decode_batch_size: int = DECODE_BATCH_SIZE):
        self.runtime = runtime
        self.hparams = hparams
        self.retokenize = bool(retokenize)
        self.decode_batch_size = max(1, int(decode_batch_size))
        self._lock = threading.RLock()
        self._threads: list[threading.Thread] = []
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
            _debug_print_control_points(
                "render_request_snapshot",
                control_points,
                extra=f"job_id={job_id}, grid={int(grid_resolution)}, sigma={float(sigma):.4f}, status={self._status}",
            )

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
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()

    def cancel_render(self):
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()

    def shutdown(self, timeout: float = 30.0):
        """Cancel the running render and wait until its worker threads are gone (so the model can be released)."""
        self.cancel_render()
        with self._lock:
            threads = list(self._threads)
        deadline = time.time() + float(timeout)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.time()))

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

    def _normalize_tokens(self, frames: np.ndarray, target_len: int) -> np.ndarray:
        """Frame sequence -> exactly target_len frames [L, 4] (PAD frames appended, or cut with EOS last)."""
        return compound.fit_frames(frames, target_len)

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
        sequence_len = int(self.hparams.max_seq_len)
        sequences = []
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
            grid_n = int(grid_resolution)
            decode_batch_size = max(1, int(self.decode_batch_size))
            chunk_latents: list[np.ndarray] = []

            def _flush_chunk():
                nonlocal completed
                if not chunk_latents:
                    return
                matrices, token_lists = self.runtime.decode_latents_to_matrices(
                    np.stack(chunk_latents, axis=0), return_tokens=True, batch_size=decode_batch_size
                )
                for matrix, original_tokens in zip(matrices, token_lists):
                    if self.retokenize:
                        tokens = self.runtime._tokens_from_matrix_buffer(matrix)
                    else:
                        tokens = original_tokens
                    sequences.append(self._normalize_tokens(tokens, sequence_len))
                completed += len(chunk_latents)
                chunk_latents.clear()
                self._publish_update(job_id, "rendering", completed, total)

            # Latents/BPMs are computed cell by cell in row-major order (the "last valid latent" fallback
            # depends on this order). The expensive decoding is batched over blocks of cells.
            for row in range(grid_n):
                wy = -field_half_extent + (row + 0.5) * cell_size
                for col in range(grid_n):
                    if cancel_event.is_set():
                        self._publish_update(job_id, "cancelled", completed, total)
                        return

                    wx = -field_half_extent + (col + 0.5) * cell_size
                    cursor = np.asarray([wx, wy], dtype=np.float32)

                    chunk_latents.append(_cell_latent(cursor))
                    bpms.append(_cell_bpm(cursor))

                    if len(chunk_latents) >= decode_batch_size:
                        _flush_chunk()
                        if cancel_event.is_set():
                            self._publish_update(job_id, "cancelled", completed, total)
                            return

            _flush_chunk()

            timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = project_root() / "outputs" / branch_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"field_render_{timestamp}.npz"

            info = compound.format_info(self.hparams)
            sequences_array = (np.stack(sequences, axis=0).astype(np.int16) if sequences
                               else np.zeros((0, sequence_len, compound.N_FIELDS), dtype=np.int16))
            np.savez_compressed(
                str(output_path),
                # compound frames [N, L, 4] = (step, pitch, velocity, deviation), see modules/compound.py
                sequences=sequences_array,
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
                # Format metadata required for external frame -> MIDI decoding.
                token_format=str(info["token_format"]),
                field_names=np.asarray(info["field_names"]),
                field_vocab_sizes=np.asarray(info["field_vocab_sizes"], dtype=np.int64),
                n_specials=int(info["n_specials"]),
                pad_token=int(info["pad_token"]),
                sos_token=int(info["sos_token"]),
                eos_token=int(info["eos_token"]),
                center_timing=int(info["center_timing"]),
                n_pitch=int(self.hparams.n_pitch),
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
    """Real-time decoder for the cursor position (small worker pool, newest finished result wins).

    request_decode() never cancels a running job. Up to `max_workers` jobs run in parallel; while all workers are
    busy only the newest request is kept (older waiting requests are dropped, they were never started). A finished
    job publishes its matrix only if no job with a higher id has published yet, so the displayed reconstruction
    never jumps back to an older latent and is replaced atomically once a newer sequence is complete.
    """

    def __init__(self, decoder: torch.nn.Module, hparams, max_workers: int | None = None, stream_partial: bool | None = None):
        self.decoder = decoder
        self.hparams = hparams
        self.device = next(decoder.parameters()).device
        self.max_len = int(hparams.max_seq_len)
        self.max_workers = max(1, int(DECODE_WORKERS if max_workers is None else max_workers))
        self.stream_partial = bool(DECODE_STREAM if stream_partial is None else stream_partial)
        self._pending: tuple[int, np.ndarray] | None = None
        self._active_workers = 0
        self._published_id = 0
        self._closed = False
        self._latest_decode_ms = 0.0
        self._tls = threading.local()

        self._lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._current_job_id = 0
        self._cancel_event = None
        self._latest_generation = 0
        self._last_consumed_generation = -1

        cols = int(self.hparams.width * self.hparams.n_timing)
        self._latest_matrix = np.zeros((int(self.hparams.height), cols), dtype=np.float32)
        self._latest_progress_step = -1
        self._latest_status = "idle"

    def request_decode(self, latent_vector: np.ndarray, seed_matrix: np.ndarray | None = None):
        """Queue a decode of this latent (seed_matrix is unused, kept for API compatibility)."""
        latent_vector = np.asarray(latent_vector, dtype=np.float32).copy()
        spawn = False
        with self._lock:
            if self._closed:
                return
            self._current_job_id += 1
            self._pending = (self._current_job_id, latent_vector)  # replaces an older request that never started
            self._latest_status = "decoding"
            if self._active_workers < self.max_workers:
                self._active_workers += 1
                spawn = True
        if spawn:
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            with self._lock:
                self._threads = [t for t in self._threads if t.is_alive()] + [thread]
            thread.start()

    def discard(self):
        """Forget the waiting request and drop the results of all jobs that are still running (decode switched off)."""
        with self._lock:
            self._pending = None
            # +1: jobs with an id up to the current one are dropped; the next request gets id current + 1 and may publish.
            self._published_id = max(self._published_id, self._current_job_id + 1)

    def shutdown(self, timeout: float = 30.0):
        """Stop accepting work and wait until the worker threads are gone (so the model can be released)."""
        with self._lock:
            self._closed = True
            self._pending = None
            threads = list(self._threads)
        deadline = time.time() + float(timeout)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.time()))

    def consume_latest_update(self):
        with self._lock:
            if self._latest_generation == self._last_consumed_generation:
                return None
            self._last_consumed_generation = self._latest_generation
            return {
                "matrix": self._latest_matrix.copy(),
                "progress_step": int(self._latest_progress_step),
                "status": self._latest_status,
                "decode_ms": float(self._latest_decode_ms),
            }

    def _publish_update(self, job_id: int, matrix: np.ndarray | None, progress_step: int, status: str,
                        decode_ms: float | None = None) -> bool:
        with self._lock:
            if job_id < self._published_id:
                return False  # a newer result is already shown
            self._published_id = job_id
            if matrix is not None:
                self._latest_matrix = matrix.astype(np.float32, copy=True)
            self._latest_progress_step = int(progress_step)
            self._latest_status = status
            if decode_ms is not None:
                self._latest_decode_ms = float(decode_ms)
            self._latest_generation += 1
            return True

    def _stream_context(self):
        """One CUDA stream per worker thread, so the parallel decodes do not wait for each other's syncs."""
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        stream = getattr(self._tls, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._tls.stream = stream
        return torch.cuda.stream(stream)

    def _worker_loop(self):
        while True:
            with self._lock:
                if self._closed or self._pending is None:
                    self._active_workers -= 1
                    return
                job_id, latent = self._pending
                self._pending = None
            try:
                with self._stream_context():
                    if self.stream_partial:
                        self._decode_job_streaming(job_id, latent)
                    else:
                        self._decode_job(job_id, latent)
            except Exception:
                traceback.print_exc()
                self._publish_update(job_id, None, -1, "error")

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

    def _decode_job(self, job_id: int, latent_vector: np.ndarray):
        """Greedy constrained decoding of one sequence (KV-cache, a sync only every few steps); the finished matrix
        replaces the displayed one in one piece (same result as the batched full-field render)."""
        h = self.hparams
        eos_token = int(h.eos_token)
        t0 = time.perf_counter()
        z = torch.from_numpy(latent_vector).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            generated = self.decoder.generate(z, max_len=self.max_len, sample=False, constrained=True,
                                              pad_after_eos=True, check_every=DECODE_CHECK_EVERY)
        row = generated[0].detach().cpu().numpy().astype(np.int64)  # the only full sync of the job
        eos_pos = np.flatnonzero(row[:, compound.FIELD_STEP] == eos_token)
        seq = row[: int(eos_pos[0]) + 1] if eos_pos.size else row
        matrix = compound.frames_to_matrix(seq, h)
        notes = seq[seq[:, compound.FIELD_STEP] >= compound.N_SPECIALS]
        progress = int(notes[-1, compound.FIELD_STEP]) - compound.N_SPECIALS if len(notes) else 0
        self._publish_update(job_id, matrix, progress, "done", decode_ms=(time.perf_counter() - t0) * 1000.0)

    def _decode_job_streaming(self, job_id: int, latent_vector: np.ndarray):
        """Like _decode_job, but publishes after every generated note (live update). A job stops as soon as a newer
        request exists, because its partial updates would otherwise fight with the newer job's."""
        h = self.hparams
        eos_token = int(h.eos_token)
        sos_token = int(h.sos_token)
        t0 = time.perf_counter()
        with self._lock:
            seed_matrix = self._latest_matrix.copy()

        partial_matrix = np.zeros_like(seed_matrix, dtype=np.float32)
        display_matrix = seed_matrix.copy()

        z = torch.from_numpy(latent_vector).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            cache = self.decoder.init_cache(z)
        cur_frame = torch.full((1, compound.N_FIELDS), sos_token, dtype=torch.long, device=self.device)

        step_idx = 0
        for _ in range(self.max_len):
            with self._lock:
                if job_id < self._current_job_id or job_id < self._published_id:
                    return  # superseded

            if cache['pos'] >= self.decoder.max_seq_len:
                break

            with torch.no_grad():
                logits = self.decoder.decode_step(cur_frame, cache)
                cur_frame = self.decoder.pick_frame(logits, cur_frame, sample=False, constrained=True)

            frame = cur_frame[0].detach().cpu().numpy()
            if int(frame[compound.FIELD_STEP]) == eos_token:
                break
            frontier = compound.write_frame(partial_matrix, frame, h)
            if frontier is None:
                continue
            step_idx = int(frame[compound.FIELD_STEP]) - compound.N_SPECIALS
            display_matrix[:, :frontier] = partial_matrix[:, :frontier]
            if not self._publish_update(job_id, display_matrix, step_idx, "decoding"):
                return

        display_matrix[:, :] = partial_matrix
        self._publish_update(job_id, display_matrix, step_idx, "done", decode_ms=(time.perf_counter() - t0) * 1000.0)


class CudaRBFRuntime:
    def __init__(self, previous: int = 0, checkpoint_kind: str | None = None, retokenize: bool = False):
        if torch.cuda.is_available():
            # With several GPUs the last one is reserved for inference (training runs on the others).
            self.device = inference_device()
            device_name = f"CUDA ({torch.cuda.get_device_name(self.device)})"
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

    def _tokens_to_matrix(self, frames) -> np.ndarray:
        """Generated compound frames [L, 4] -> note matrix (see modules.compound.frames_to_matrix)."""
        return compound.frames_to_matrix(frames, self.hparams)

    def decode_latents_to_matrices(
        self,
        latent_vectors: np.ndarray,
        return_tokens: bool = False,
        batch_size: int = 64,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        """Decode many latent vectors at once (batched greedy decoding with KV-cache).

        Returns one matrix per latent (and, optionally, the generated frames [n, 4] up to and including EOS).
        """
        latents = np.asarray(latent_vectors, dtype=np.float32)
        if latents.ndim == 1:
            latents = latents.reshape(1, -1)
        eos_token = int(self.hparams.eos_token)
        max_len = int(self.hparams.max_seq_len)
        batch_size = max(1, int(batch_size))

        matrices: list[np.ndarray] = []
        token_lists: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, latents.shape[0], batch_size):
                z = torch.from_numpy(latents[start:start + batch_size]).to(self.device)
                generated = self.decoder.generate(z, max_len=max_len, sample=False, constrained=True, pad_after_eos=True,
                                                  check_every=DECODE_CHECK_EVERY)
                generated = generated.detach().cpu().numpy().astype(np.int64)  # [B, L, 4]
                for row in generated:
                    eos_pos = np.flatnonzero(row[:, compound.FIELD_STEP] == eos_token)
                    seq = row[: int(eos_pos[0]) + 1] if eos_pos.size else row
                    token_lists.append(seq)
                    matrices.append(self._tokens_to_matrix(seq))

        if return_tokens:
            return matrices, token_lists
        return matrices

    def decode_latent_to_matrix(self, latent_vector: np.ndarray, return_tokens: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        latent = np.asarray(latent_vector, dtype=np.float32).reshape(1, -1)
        if return_tokens:
            matrices, token_lists = self.decode_latents_to_matrices(latent, return_tokens=True, batch_size=1)
            return matrices[0], token_lists[0]
        return self.decode_latents_to_matrices(latent, batch_size=1)[0]

    def _tokens_from_matrix_buffer(self, matrix_buffer: np.ndarray) -> np.ndarray:
        """Note matrix -> compound frames [K + 1, 4] (notes sorted by (step, pitch), EOS last)."""
        try:
            return compound.matrix_to_frames(np.asarray(matrix_buffer, dtype=np.float32), self.hparams)
        except ValueError as exc:
            raise ValueError(f"midi_buffer.{exc}")

    def _prepare_token_batch(self, raw_frames: np.ndarray) -> torch.Tensor:
        """Frames [K, 4] -> [1, max_seq_len, 4] (PAD frames appended, or cut with EOS last)."""
        frames = compound.fit_frames(raw_frames, int(self.hparams.max_seq_len)).astype(np.int64)
        return torch.from_numpy(frames).to(self.device, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def _frames_from_list(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[-1] != compound.N_FIELDS:
            raise ValueError(
                "midi_buffer frames must be a list of [step, pitch, velocity, deviation] frames "
                "(flat triplet token lists are no longer supported)."
            )
        return arr

    def encode_midi_buffer_to_latent(self, midi_buffer) -> np.ndarray:
        if midi_buffer is None:
            raise ValueError("midi_buffer is missing while Encode MIDI is enabled.")
        if isinstance(midi_buffer, dict) and "frames" in midi_buffer:
            raw_frames = self._frames_from_list(midi_buffer["frames"])
        elif isinstance(midi_buffer, dict) and "tokens" in midi_buffer:
            raw_frames = self._frames_from_list(midi_buffer["tokens"])
        elif isinstance(midi_buffer, dict) and "matrix" in midi_buffer:
            raw_frames = self._tokens_from_matrix_buffer(np.asarray(midi_buffer["matrix"], dtype=np.float32))
        elif isinstance(midi_buffer, list):
            raw_frames = self._frames_from_list(midi_buffer)
        else:
            raise ValueError("midi_buffer must be a frame list [[s, p, v, d], ...], {frames:[...]}, or {matrix:[[...], ...]}.")
        if raw_frames.size and not np.any(raw_frames[:, compound.FIELD_STEP] == int(self.hparams.eos_token)):
            raw_frames = np.concatenate([raw_frames, compound.special_frame("eos").reshape(1, -1).astype(np.int64)], axis=0)
        if raw_frames.size == 0:
            raw_frames = compound.special_frame("eos").reshape(1, -1).astype(np.int64)

        frames_batch = self._prepare_token_batch(raw_frames)
        with torch.no_grad():
            mu, _ = self.encoder(frames_batch)
        return mu.squeeze(0).detach().cpu().numpy().astype(np.float32)


TRAIN_WORKER_PATH = Path(__file__).resolve().parent / "train_worker.py"
TRAIN_RESUME_MODES = ("best_val", "best_train", "last", "none")

# key -> (min, max, min_inclusive, max_inclusive)
_TRAIN_RANGES = {
    "validation_split": (0.0, 1.0, True, False),
    "batch_size": (1, 65536, True, True),
    "n_epochs": (1, 100000, True, True),
    "learning_rate": (0.0, 1.0, False, True),
    "scheduler_start_lr": (0.0, 1.0, False, True),
    "scheduler_end_lr": (0.0, 1.0, True, True),
    "scheduler_exp_curve_factor": (-50.0, 50.0, True, True),
    "scheduler_warmup_ratio": (0.0, 0.5, True, True),
    "last_checkpoint_interval": (1, 100000, True, True),
    "early_stopping_patience": (1, 100000, True, True),
    "early_stopping_min_delta": (0.0, 100.0, True, True),
    "max_seq_len": (2, 4096, True, True),
    "weight_decay": (0.0, 10.0, True, True),
    "n_samples": (0, 10 ** 9, True, True),
    "free_bits": (0.0, 100.0, True, True),
    "annealing_slope": (0.0, 1.0, True, True),
    "seed": (0, 2 ** 32 - 1, True, True),
    "d_model": (8, 8192, True, True),
    "n_heads": (1, 64, True, True),
    "n_layers": (1, 64, True, True),
    "ff_dim": (1, 65536, True, True),
    "dropout": (0.0, 1.0, True, False),
    "latent_dim": (1, 8192, True, True),
    "n_memory_tokens": (1, 512, True, True),
}


def validate_train_config(raw, available_datasets, checkpoint_count, defaults, legacy_datasets=(),
                          gpu_count=None) -> tuple[dict, dict]:
    """Validate a TrainConfig coming from the browser.

    gpu_count: number of CUDA devices (None = ask torch). The training GPUs are chosen automatically
    (modules.distributed.device_roles: with 2+ GPUs the last one is kept for inference, the others train; with more
    than one training GPU the training runs with DistributedDataParallel). options.gpus from the client is ignored.

    available_datasets are the compound datasets of the branch; legacy_datasets (old triplet format) are rejected
    with a field error.

    Returns (clean_config, errors). errors maps a field name (or "general") to a message; the config may only be
    used when errors is empty. checkpoint_count(mode) -> number of checkpoints available for a resume mode.
    """
    errors: dict[str, str] = {}
    raw = raw if isinstance(raw, dict) else {}
    defaults = defaults or {}

    mode = str(raw.get("mode", "new")).lower()
    if mode not in ("new", "resume"):
        errors["mode"] = "must be 'new' or 'resume'"
        mode = "new"
    clean: dict = {"mode": mode, "overrides": {}, "options": {}}

    raw_overrides = raw.get("overrides") or {}
    if not isinstance(raw_overrides, dict):
        errors["general"] = "overrides must be an object"
        raw_overrides = {}

    overrides = clean["overrides"]
    for key, value in raw_overrides.items():
        if key not in ALLOWED_KEYS:
            errors[key] = "unknown parameter"
            continue
        if mode == "resume" and key in ARCHITECTURE_KEYS:
            continue  # architecture comes from the checkpoint
        try:
            overrides[key] = coerce_param(key, value)
        except ValueError as exc:
            errors[key] = str(exc)

    for key, (lo, hi, lo_incl, hi_incl) in _TRAIN_RANGES.items():
        if key not in overrides or key in errors:
            continue
        value = overrides[key]
        if (value < lo if lo_incl else value <= lo) or (value > hi if hi_incl else value >= hi):
            left = "[" if lo_incl else "("
            right = "]" if hi_incl else ")"
            errors[key] = f"must be in {left}{lo}, {hi}{right}"

    points = overrides.get("annealing_list")
    if points is not None and "annealing_list" not in errors:
        if len(points) < 2:
            errors["annealing_list"] = "needs at least two points"
        elif any(not (0.0 <= x <= 1.0) for x, _ in points):
            errors["annealing_list"] = "positions must be within [0, 1]"
        elif any(points[i][0] >= points[i + 1][0] for i in range(len(points) - 1)):
            errors["annealing_list"] = "positions must be strictly ascending"
        elif any(y < 0.0 or y > 100.0 for _, y in points):
            errors["annealing_list"] = "beta values must be in [0, 100]"

    if mode == "new":
        d_model = overrides.get("d_model", defaults.get("d_model"))
        n_heads = overrides.get("n_heads", defaults.get("n_heads"))
        if d_model and n_heads and "d_model" not in errors and "n_heads" not in errors and int(d_model) % int(n_heads) != 0:
            errors["d_model"] = f"must be divisible by n_heads ({n_heads})"
            errors["n_heads"] = f"must divide d_model ({d_model})"

    legacy_datasets = list(legacy_datasets or ())
    if "dataset_name" in overrides:
        wanted = overrides["dataset_name"]
    else:
        wanted = defaults.get("dataset_name") if mode == "new" else None
    if wanted is not None and wanted in legacy_datasets and "dataset_name" not in errors:
        errors["dataset_name"] = f"'{wanted}' is a legacy triplet dataset, re-extract it in the Dataset tab"
    if not available_datasets:
        if legacy_datasets:
            errors["general"] = (f"No compound dataset for this branch: {len(legacy_datasets)} legacy (triplet) "
                                 f"dataset(s) are blocked, re-extract them in the Dataset tab.")
        else:
            errors["general"] = "No dataset found for this branch (data/<branch>/*.npz)."
    elif wanted is not None and wanted not in available_datasets and "dataset_name" not in errors:
        errors["dataset_name"] = f"dataset '{wanted}' not found"

    resume_raw = raw.get("resume") or {}
    if mode == "resume":
        if not isinstance(resume_raw, dict):
            errors["general"] = "resume must be an object"
            resume_raw = {}
        resume_mode = str(resume_raw.get("mode", "last")).lower()
        if resume_mode not in TRAIN_RESUME_MODES:
            errors["resume_mode"] = f"must be one of {', '.join(TRAIN_RESUME_MODES)}"
            resume_mode = "last"
        try:
            previous = coerce_param_number(resume_raw.get("previous", 0), integer=True)
        except ValueError as exc:
            errors["resume_previous"] = str(exc)
            previous = 0
        if previous < 0:
            errors["resume_previous"] = "must be >= 0"
        resume_clean = {"mode": resume_mode, "previous": previous, "continue_beta": bool(resume_raw.get("continue_beta", False))}
        for key, field in (("start_lr", "resume_start_lr"), ("end_lr_factor", "resume_end_lr_factor")):
            if resume_raw.get(key) not in (None, ""):
                try:
                    number = coerce_param_number(resume_raw[key])
                    if number <= 0.0:
                        raise ValueError("must be > 0")
                    resume_clean[key] = number
                except ValueError as exc:
                    errors[field] = str(exc)
        if "resume_mode" not in errors and "resume_previous" not in errors:
            if checkpoint_count(resume_mode) <= previous:
                errors["resume_mode"] = f"no '{resume_mode}' checkpoint #{previous} available"
        clean["resume"] = resume_clean

    options_raw = raw.get("options") or {}
    options_raw = options_raw if isinstance(options_raw, dict) else {}
    clean["options"]["torch_compile"] = bool(options_raw.get("torch_compile", True))
    if gpu_count is None:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpus: list[int] = list(device_roles(int(gpu_count))["train_gpus"])
    clean["options"]["gpus"] = gpus
    if len(gpus) > 1:
        batch = overrides.get("batch_size", defaults.get("batch_size"))
        if batch is not None and "batch_size" not in errors and int(batch) < len(gpus):
            errors["batch_size"] = f"must be at least the number of training GPUs ({len(gpus)}); it is split over them"
    try:
        log_interval = coerce_param_number(options_raw.get("log_interval", 25), integer=True)
        if not (1 <= log_interval <= 1000):
            raise ValueError("must be in [1, 1000]")
        clean["options"]["log_interval"] = log_interval
    except ValueError as exc:
        errors["log_interval"] = str(exc)

    return clean, errors


def coerce_param_number(value, integer: bool = False):
    """Small helper for options that are not part of the hparams whitelist."""
    if isinstance(value, bool):
        raise ValueError("must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("must be a number")
    if not math.isfinite(number):
        raise ValueError("must be finite")
    if integer:
        if abs(number - round(number)) > 1e-9:
            raise ValueError("must be an integer")
        return int(round(number))
    return number


class TrainingManager:
    """Runs one training in a subprocess (train_worker.py) and keeps its progress for late-joining clients.

    States: idle | starting | running | stopping | finished | failed.
    The worker writes one JSON event per stdout line; a reader thread folds them into this object.
    `version` increases with every change so the broadcast loop only emits when something happened.
    """

    ACTIVE_STATES = ("starting", "running", "stopping")

    def __init__(self, on_finish=None, worker_path: Path | None = None, max_epochs: int = 5000, max_logs: int = 400, max_batches: int = 600):
        self._lock = threading.RLock()
        self._on_finish = on_finish
        self._worker_path = Path(worker_path) if worker_path else TRAIN_WORKER_PATH
        self._max_epochs = int(max_epochs)
        self._max_logs = int(max_logs)
        self._max_batches = int(max_batches)
        self._proc: subprocess.Popen | None = None
        self._proc_alive = False
        self._finalizing = False
        self._stop_requested = False
        self._tmpdir: str | None = None
        self._stop_file: str | None = None
        self._stderr_tail: deque = deque(maxlen=40)
        self._stderr_thread: threading.Thread | None = None
        self._done_event: dict | None = None
        self._error_event: dict | None = None
        self.version = 0
        self._reset_run()

    # ------------------------------------------------------------------ state helpers
    def _reset_run(self):
        self.state = "idle"
        self.config: dict | None = None
        self.epochs: deque = deque(maxlen=self._max_epochs)
        self.batches: deque = deque(maxlen=self._max_batches)
        self.last_batch: dict | None = None
        self.logs: deque = deque(maxlen=self._max_logs)
        self.checkpoints: list[dict] = []
        self.error: str | None = None
        self.status_text = ""
        self.stopped = False
        self.pid: int | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._done_event = None
        self._error_event = None
        self._stop_requested = False
        self._stderr_tail.clear()

    def _bump(self):
        self.version += 1

    def _append_log(self, text: str):
        self.logs.append({"t": time.time(), "text": str(text)})

    def is_active(self) -> bool:
        """True while the worker process exists (inference must stay paused)."""
        with self._lock:
            return bool(self._proc_alive)

    # ------------------------------------------------------------------ start / stop
    def start(self, config: dict, before_spawn=None):
        with self._lock:
            if self._proc_alive or self._finalizing:
                raise RuntimeError("A training is already running.")
            self._reset_run()
            self._proc_alive = True
            self.state = "starting"
            self.config = config
            self.started_at = time.time()
            self.status_text = "starting"
            self._bump()

        try:
            if before_spawn is not None:
                before_spawn()
            tmpdir = tempfile.mkdtemp(prefix="rbf_train_")
            config_path = os.path.join(tmpdir, "config.json")
            stop_file = os.path.join(tmpdir, "STOP")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [sys.executable, str(self._worker_path), "--config", config_path, "--stop-file", stop_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(self._worker_path.parent),
                env=env,
            )
        except Exception as exc:
            traceback.print_exc()
            self._finalize(returncode=None, message=f"Failed to start training: {exc}")
            raise

        with self._lock:
            self._proc = proc
            self._tmpdir = tmpdir
            self._stop_file = stop_file
            self.pid = proc.pid
            stop_now = self._stop_requested
        self._stderr_thread = threading.Thread(target=self._read_stderr, args=(proc,), daemon=True)
        self._stderr_thread.start()
        threading.Thread(target=self._read_stdout, args=(proc,), daemon=True).start()
        if stop_now:
            self._signal_stop(proc, stop_file)

    def _signal_stop(self, proc, stop_file):
        if stop_file:
            try:
                Path(stop_file).touch()
            except OSError:
                pass
        if proc is not None and os.name != "nt":
            try:
                proc.send_signal(signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

    def stop(self, force: bool = False) -> bool:
        """Ask the worker to stop after the current batch. force=True kills the process immediately."""
        with self._lock:
            if not self._proc_alive:
                return False
            self._stop_requested = True
            if self.state in ("starting", "running"):
                self.state = "stopping"
                self.status_text = "stopping"
                self._bump()
            proc = self._proc
            stop_file = self._stop_file
        if force and proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
        else:
            self._signal_stop(proc, stop_file)
        return True

    def terminate(self, timeout: float = 30.0):
        """Used at server shutdown: stop the worker and make sure it does not outlive the server."""
        if not self.stop():
            return
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass

    # ------------------------------------------------------------------ reader threads
    def _read_stderr(self, proc):
        try:
            for line in proc.stderr:
                text = line.rstrip("\n")
                if text:
                    self._stderr_tail.append(text)
                    print(f"[train] {text}", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            pass

    def _read_stdout(self, proc):
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    with self._lock:
                        self._append_log(line)
                        self._bump()
                    continue
                if isinstance(event, dict):
                    self._handle_event(event)
        except (ValueError, OSError):
            pass
        finally:
            try:
                returncode = proc.wait()
            except Exception:
                returncode = None
            self._finalize(returncode=returncode)

    def _handle_event(self, event: dict):
        kind = event.get("type")
        with self._lock:
            if self.state == "starting":
                self.state = "running"
            if kind == "status":
                self.status_text = str(event.get("text", ""))
                self._append_log(self.status_text)
            elif kind == "batch":
                self.last_batch = event
                self.batches.append({k: event.get(k) for k in ("epoch", "step", "steps", "loss", "beta", "lr")})
            elif kind == "epoch":
                self.epochs.append(event)
            elif kind == "checkpoint":
                self.checkpoints.append({"kind": event.get("kind"), "path": event.get("path"), "time": time.time()})
            elif kind == "log":
                self._append_log(event.get("text", ""))
            elif kind == "done":
                self._done_event = event
                self.stopped = bool(event.get("stopped"))
            elif kind == "error":
                self._error_event = event
                self._append_log(f"ERROR: {event.get('message', '')}")
            self._bump()

    def _finalize(self, returncode, message: str | None = None):
        with self._lock:
            if self._finalizing:
                return
            self._finalizing = True
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)

        error = None
        with self._lock:
            if self._error_event is not None:
                error = str(self._error_event.get("message") or "Training failed")
            elif self._done_event is None:
                if message:
                    error = message
                elif returncode not in (0, None):
                    if self._stop_requested:
                        self.stopped = True
                        self.status_text = "terminated"
                    else:
                        tail = " | ".join(list(self._stderr_tail)[-3:])
                        error = f"Training process exited with code {returncode}" + (f": {tail}" if tail else "")
            final_state = "failed" if error else "finished"
            self._proc_alive = False  # inference may come back from here on
            tmpdir = self._tmpdir
            self._tmpdir = None

        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if self._on_finish is not None:
            try:
                self._on_finish(final_state)
            except Exception:
                traceback.print_exc()

        with self._lock:
            self.state = final_state
            self.error = error
            self.finished_at = time.time()
            self._proc = None
            self._finalizing = False
            self._bump()

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            elapsed = None
            if self.started_at is not None:
                elapsed = (self.finished_at or now) - self.started_at
            return {
                "state": self.state,
                "active": self.state in self.ACTIVE_STATES,
                "stopped": bool(self.stopped),
                "status_text": self.status_text,
                "config": self.config,
                "epochs": list(self.epochs),
                "batches": list(self.batches),
                "last_batch": self.last_batch,
                "logs": list(self.logs),
                "checkpoints": list(self.checkpoints),
                "error": self.error,
                "pid": self.pid,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed": elapsed,
                "version": self.version,
            }


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
        return project_root() / "data" / branch_name

    def _dataset_formats(self) -> dict[str, str]:
        """name -> token format of every .npz in data/<branch>/ (cheap header read)."""
        data_dir = self._data_branch_dir()
        if not data_dir.exists():
            return {}
        names = sorted([p.name for p in data_dir.glob("*.npz") if p.is_file()])
        return {name: dataset_tools.dataset_format_of(name)[0] for name in names}

    def _list_available_datasets(self) -> list[str]:
        """Only compound datasets can be used for training and inference."""
        return [name for name, fmt in self._dataset_formats().items() if fmt == dataset_tools.compound.TOKEN_FORMAT]

    def _list_legacy_datasets(self) -> list[str]:
        return [name for name, fmt in self._dataset_formats().items() if fmt != dataset_tools.compound.TOKEN_FORMAT]

    def _resolve_dataset_name(self, preferred: str | None) -> str:
        options = self._list_available_datasets()
        if not options:
            legacy = self._list_legacy_datasets()
            if legacy:
                raise FileNotFoundError(
                    f"No compound dataset in {self._data_branch_dir()}: {len(legacy)} legacy (triplet) dataset(s) "
                    f"are blocked, re-extract them in the Dataset tab"
                )
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
        if len(control_points) == 0 or bpm_targets.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.float32)
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
        model_dir = project_root() / "models" / branch_name
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
        self._last_shuffle_nonce = 0  # last 'reshuffle samples' request of the client that was handled
        self.sample_count_max = 40
        self.selected_mode = "none"
        self.selected_previous = 0

        # The app also starts without a model and/or dataset: it then runs idle and reports what is missing.
        self.hparams = SimpleNamespace(**vars(_DEFAULT_HPARAMS))
        self.runtime = None
        self.decoder_manager = None
        self.render_manager = None
        self.model_error: str | None = None
        self.dataset_error: str | None = None
        self.dataset_name = ""
        self.latent_vectors = np.zeros((0, int(getattr(self.hparams, "latent_dim", 0) or 0)), dtype=np.float32)
        self.labels: list[str] = []
        self.anchor_bpms = np.zeros(0, dtype=np.float32)
        # Original notes of every anchor as raw [step, pitch, velocity, deviation] lists (for the reconstruction viewer).
        self.anchor_notes: list[list[list[int]]] = []
        self.pool_version = 0
        self._lock = threading.Lock()
        self._train_start_lock = threading.Lock()
        self.training = TrainingManager(on_finish=self._on_training_finished)

        self._try_init_runtime()
        self._refresh_dataset(getattr(self.hparams, "dataset_name", None))

        self.model_catalog = self._build_model_catalog()
        self._normalize_selection_to_catalog()

        self._load_latent_pool()
        control_points = self._make_initial_control_points(len(self.latent_vectors), field_half_extent=2.0)
        self.field_half_extent = 2.0
        self.weights = _safe_rbf_weights(control_points, self.latent_vectors, sigma=0.30)
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
        return project_root() / "models" / branch_name

    def _try_init_runtime(self) -> bool:
        """Try to load a model checkpoint. Never raises; on failure the app stays idle and model_error is set."""
        self.model_error = None
        selected_mode = None
        for candidate in ["none", "best_val", "best_train", "last"]:
            if self._mode_files_static(candidate):
                selected_mode = candidate
                break
        if selected_mode is None:
            self.model_error = f"No model checkpoints found in {self._branch_model_dir()}"
            print(f"[warning] {self.model_error}. Starting without a model (idle).", flush=True)
            return False

        self.selected_mode = selected_mode
        self.selected_previous = 0
        checkpoint_kind = None if selected_mode == "none" else selected_mode
        try:
            runtime = CudaRBFRuntime(previous=0, checkpoint_kind=checkpoint_kind, retokenize=RETOKENIZE)
        except Exception as exc:
            traceback.print_exc()
            self.model_error = f"Failed to load model checkpoint: {exc}"
            return False

        self.runtime = runtime
        self.hparams = runtime.hparams
        self.decoder_manager = RealtimeDecoderManager(runtime.decoder, self.hparams)
        self.render_manager = FullFieldRenderManager(runtime, self.hparams, retokenize=RETOKENIZE)
        return True

    def _refresh_dataset(self, preferred: str | None = None):
        """Resolve the dataset to use. Never raises; without datasets dataset_name is empty and dataset_error is set."""
        try:
            self.dataset_name = self._resolve_dataset_name(preferred)
            self.dataset_error = None
        except FileNotFoundError as exc:
            self.dataset_name = ""
            self.dataset_error = str(exc)
            print(f"[warning] {exc}. Starting without a dataset (idle).", flush=True)
        self.hparams.dataset_name = self.dataset_name

    def _app_status(self) -> dict:
        model_ok = self.runtime is not None
        dataset_ok = bool(self.dataset_name) and len(self.latent_vectors) > 0
        training = self.training.is_active()
        messages: list[str] = []
        # Inference stays available while a training runs (own GPU if there are several, else the shared device).
        if not model_ok:
            messages.append(self.model_error or "No model loaded.")
        if not dataset_ok:
            messages.append(self.dataset_error or "No dataset loaded.")
        return {
            "ready": bool(model_ok and dataset_ok),
            "model_available": bool(model_ok),
            "dataset_available": bool(dataset_ok),
            "training": bool(training),
            "messages": messages,
        }

    def _is_idle(self) -> bool:
        return not self._app_status()["ready"]

    def _model_info(self) -> dict:
        """Most important parameters of the active model for the GUI (empty without a model)."""
        runtime = self.runtime
        if runtime is None:
            return {}
        cached = getattr(self, "_model_info_cache", None)
        if cached is not None and cached[0] is runtime:
            return cached[1]
        hp = runtime.hparams
        try:
            n_params = sum(p.numel() for p in runtime.model.parameters())
        except Exception:
            n_params = 0
        info = {
            "parameters": f"{n_params / 1e6:.2f} M" if n_params else "?",
            "d_model": getattr(hp, "d_model", "?"),
            "layers": getattr(hp, "n_layers", "?"),
            "heads": getattr(hp, "n_heads", "?"),
            "ff_dim": getattr(hp, "ff_dim", "?"),
            "latent_dim": getattr(hp, "latent_dim", "?"),
            "memory_tokens": getattr(hp, "n_memory_tokens", "?"),
            "max_seq_len": getattr(hp, "max_seq_len", "?"),
            "norm": "pre-norm" if bool(getattr(hp, "norm_first", False)) else "post-norm",
            "dropout": getattr(hp, "dropout", "?"),
            "format": getattr(hp, "token_format", "?"),
            "trained_on": getattr(hp, "dataset_name", "") or "?",
            "device": str(runtime.device),
        }
        epochs = getattr(hp, "n_epochs", None)
        if epochs:
            info["epochs"] = epochs
        self._model_info_cache = (runtime, info)
        return info

    def recheck_resources(self):
        """Retry loading model/dataset (e.g. after files were added or a training wrote new checkpoints)."""
        with self._lock:
            had_pool = self.runtime is not None and len(self.latent_vectors) > 0
            loaded_mode = self.selected_mode
            loaded_stamp = self._selected_timestamp() if self.runtime is not None else None
            if self.runtime is None:
                self._try_init_runtime()
                loaded_mode, loaded_stamp = self.selected_mode, None
            self._refresh_dataset(self.dataset_name or getattr(self.hparams, "dataset_name", None))
            self.model_catalog = self._build_model_catalog()
            if loaded_stamp is not None:
                # New checkpoints (e.g. written by a running training) shift the 'previous' indices: keep pointing at
                # the checkpoint that is actually loaded.
                items = self.model_catalog.get("modes", {}).get(loaded_mode, {}).get("items", [])
                for item in items:
                    if item.get("timestamp") == loaded_stamp:
                        self.selected_mode = loaded_mode
                        self.selected_previous = int(item["previous"])
                        break
            self._normalize_selection_to_catalog()
            if not had_pool and self.runtime is not None and self.dataset_name:
                self._load_latent_pool()
                self._rebuild_pool_dependent_state(reset_modes=True)

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

    def _set_empty_pool(self):
        dim = int(getattr(self.hparams, "latent_dim", 0) or 0)
        self.latent_vectors = np.zeros((0, dim), dtype=np.float32)
        self.labels = []
        self.anchor_bpms = np.zeros(0, dtype=np.float32)
        self.anchor_notes = []
        self.pool_version += 1
        # Keep the requested sample count / default maximum so the UI sliders stay untouched while idle.
        self.sample_count_max = 40
        self._recompute_fallback_caches()

    def _load_latent_pool(self):
        if self.runtime is None or not self.dataset_name:
            self._set_empty_pool()
            return
        try:
            self._load_latent_pool_impl()
            self.dataset_error = None
        except Exception as exc:
            traceback.print_exc()
            self.dataset_error = f"Failed to load dataset '{self.dataset_name}': {exc}"
            self._set_empty_pool()

    def _load_latent_pool_impl(self):
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
        frames = torch.from_numpy(np.asarray(data, dtype=np.int64)).to(self.runtime.device, dtype=torch.long)  # [B, T, 4]
        with torch.no_grad():
            mu, _ = self.runtime.encoder(frames)
        self.latent_vectors = mu.detach().cpu().numpy().astype(np.float32)
        self.anchor_notes = [
            compound.frames_to_raw(row, self.hparams).astype(int).tolist()
            for row in np.asarray(data, dtype=np.int64)
        ]
        self.pool_version += 1

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
        if self.render_manager is not None:
            self.render_manager.cancel_render()
        if self.decoder_manager is not None:
            self.decoder_manager.discard()  # results of the old pool must not show up in the new one
        self.decode_ms = 0.0
        control_points = self._make_initial_control_points(len(self.latent_vectors), field_half_extent=self.field_half_extent)
        self.state.control_points = control_points
        self.state.cursor = np.zeros(2, dtype=np.float32)
        self.state.latent_vector = None
        self.state.current_bpm = float(self.anchor_bpms[0]) if len(self.anchor_bpms) > 0 else 0.0
        # Decode is on by default as soon as a model and a dataset are available.
        self.state.decode_enabled = bool(self.runtime is not None and len(self.latent_vectors) > 0)
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
        self.weights = _safe_rbf_weights(control_points, self.latent_vectors, sigma=self.state.sigma)
        self.bpm_weights = self._compute_bpm_weights(control_points, sigma=self.state.sigma)
        self._cached_heatmap = np.zeros((self.state.grid_resolution, self.state.grid_resolution), dtype=np.float32)
        _debug_print_control_points(
            "pool_rebuild",
            control_points,
            extra=f"sample_count={int(self.sample_count)}, sample_randomize={bool(self.sample_randomize)}, dataset={self.dataset_name}",
        )

    # ------------------------------------------------------------------ training integration
    def _checkpoint_count(self, mode: str) -> int:
        if mode not in self.MODE_PREFIX:
            return 0
        return len(self._mode_files_static(mode))

    def _release_device_memory(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            traceback.print_exc()

    def device_plan(self) -> dict:
        """Automatic device split between training and inference (for the GUI)."""
        gpus = cuda_device_list()
        roles = device_roles(len(gpus))
        label = _training_device_label()
        if not gpus:
            return {"mode": "shared", "train_gpus": [], "inference_gpu": None, "train_device": label,
                    "inference_device": label}
        names = {g["index"]: g["name"] for g in gpus}
        return {
            "mode": "shared" if roles["shared"] else "split",
            "train_gpus": roles["train_gpus"],
            "inference_gpu": roles["inference_gpu"],
            "train_device": ", ".join(f"GPU {i} ({names.get(i, '?')})" for i in roles["train_gpus"]),
            "inference_device": f"GPU {roles['inference_gpu']} ({names.get(roles['inference_gpu'], '?')})",
        }

    def _on_training_finished(self, final_state: str):
        """Called by the TrainingManager (reader thread) after the worker ended: refresh checkpoints / dataset list."""
        try:
            self.recheck_resources()
        except Exception:
            traceback.print_exc()

    def train_defaults(self) -> dict:
        """Current content of hparams.py (read fresh, so values saved from the GUI show up without a restart)."""
        try:
            values = read_hparams()
        except Exception:
            traceback.print_exc()
            values = dict(vars(_DEFAULT_HPARAMS))
        return {key: values[key] for key in ALLOWED_KEYS if key in values}

    def start_training(self, raw_config) -> tuple[bool, dict]:
        """Validate and start a training. Returns (ok, errors); errors maps field -> message."""
        with self._train_start_lock:
            if self.training.is_active():
                return False, {"general": "A training is already running."}
            clean, errors = validate_train_config(
                raw_config,
                self._list_available_datasets(),
                self._checkpoint_count,
                self.train_defaults(),
                legacy_datasets=self._list_legacy_datasets(),
            )
            if errors:
                return False, errors
            try:
                self.training.start(clean)
            except Exception as exc:
                return False, {"general": str(exc)}
            return True, {}

    def stop_training(self, force: bool = False) -> bool:
        return self.training.stop(force=force)

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
        runtime = CudaRBFRuntime(previous=previous, checkpoint_kind=checkpoint_kind, retokenize=RETOKENIZE)
        if self.render_manager is not None:
            self.render_manager.cancel_render()
        self.runtime = runtime
        self.model_error = None
        self.hparams = self.runtime.hparams
        self.decoder_manager = RealtimeDecoderManager(self.runtime.decoder, self.hparams)
        self.render_manager = FullFieldRenderManager(self.runtime, self.hparams, retokenize=RETOKENIZE)
        self._refresh_dataset(previous_dataset_name)
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

            preserved_points = _clamp_points_to_extent(preserved_points, self.field_half_extent)

            self.state.control_points = preserved_points.astype(np.float32)
            if previous_cursor.shape == (2,):
                self.state.cursor = _clamp_point_to_extent(previous_cursor.astype(np.float32), self.field_half_extent)

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
        if self.decoder_manager is None:
            return False
        decode_update = self.decoder_manager.consume_latest_update()
        if decode_update is None:
            return False
        self.reconstruction_matrix = decode_update["matrix"]
        self.decode_status = decode_update["status"]
        self.decode_ms = float(decode_update.get("decode_ms", 0.0) or 0.0)
        self.decode_progress_step = int(decode_update["progress_step"])
        return True

    def _refresh_render_update(self) -> bool:
        if self.render_manager is None:
            return False
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
            "n_vel": int(self.hparams.n_vel),
            "pool_version": int(self.pool_version),
            # The anchor notes are only sent when the client does not have this pool version yet.
            **({"anchor_notes": self.anchor_notes}
               if (self.state.last_frontend_payload or {}).get("anchor_notes_version") != self.pool_version else {}),
            "recon_cols": int(self.hparams.width * self.hparams.n_timing),
            "height": int(self.hparams.height),
            **_pitch_info(int(self.hparams.height)),
            "decode_status": self.decode_status,
            "decode_progress_step": int(self.decode_progress_step),
            "decode_ms": float(getattr(self, "decode_ms", 0.0)),
            "model_info": self._model_info(),
            "render_status": self.render_status,
            "render_completed": int(self.render_completed),
            "render_total": int(self.render_total),
            "render_output_path": self.render_output_path,
            "render_error": self.render_error,
            "app_status": self._app_status(),
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
            prev_frontend_payload = self.state.last_frontend_payload or {}
            prev_button_values = prev_frontend_payload.get("button_values") or {}
            prev_client_render_enabled = bool(prev_button_values.get("renderEnabled", False))
            prev_render_status = self.render_status
            self._refresh_render_update()
            self.state.last_frontend_payload = payload
            slider_values = payload.get("slider_values") or {}
            button_values = payload.get("button_values") or {}

            if self._is_idle():
                # No model and/or no dataset: nothing to interpolate or decode. Stay idle, but allow
                # switching to another dataset when a model is loaded.
                requested_dataset = str(button_values.get("datasetName", self.dataset_name) or "")
                if (
                    self.runtime is not None
                    and requested_dataset
                    and requested_dataset != self.dataset_name
                    and requested_dataset in self._list_available_datasets()
                ):
                    self.dataset_name = requested_dataset
                    self.hparams.dataset_name = requested_dataset
                    self._load_latent_pool()
                    self._rebuild_pool_dependent_state(reset_modes=False)
                if self._is_idle():
                    self.state.decode_enabled = False
                self.state.render_enabled = False
                self.reconstruction_matrix.fill(0.0)
                self.decode_status = "idle"
                self.decode_progress_step = -1
                return self._build_response(midi_buffer=None, encode_midi=False)

            prev_points_snapshot = self.state.control_points.copy()

            cursor = np.asarray(payload.get("cursor", self.state.cursor), dtype=np.float32).reshape(2)
            cursor = _clamp_point_to_extent(cursor, self.field_half_extent)
            raw_points = np.asarray(payload.get("draggable_points", self.state.control_points), dtype=np.float32)
            raw_points = _clamp_points_to_extent(raw_points, self.field_half_extent)

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
            client_render_enabled = bool(button_values.get("renderEnabled", self.state.render_enabled))
            render_enabled = client_render_enabled
            client_render_rising_edge = client_render_enabled and (not prev_client_render_enabled)
            # Guard against stale heartbeat re-arming the render when the job JUST finished
            # in THIS frame. _refresh_render_update() above may have just flipped render_status
            # to a terminal state (done/cancelled/error), but the heartbeat payload still carries
            # renderEnabled=true because the frontend hasn't received the update yet.
            # We only clamp to False on the exact transition frame (prev was "rendering" and now
            # it is terminal) so that a deliberate re-start on a subsequent frame is still allowed.
            just_completed = (
                prev_render_status == "rendering"
                and self.render_status in {"done", "cancelled", "error"}
            )
            if just_completed:
                render_enabled = False
            sample_count = _clamp_sample_count(
                _safe_int(slider_values.get("sampleCount"), self.state.sample_count),
                self.state.sample_count_max,
            )
            sample_randomize = bool(button_values.get("sampleRandomize", self.state.sample_randomize))
            dataset_name = str(button_values.get("datasetName", self.dataset_name))

            # 'Reshuffle' button: every click sends a new nonce; 0 (page load) never triggers a reshuffle.
            shuffle_nonce = _safe_int(button_values.get("sampleShuffleNonce"), 0)
            reshuffle = False
            if shuffle_nonce and shuffle_nonce != self._last_shuffle_nonce:
                self._last_shuffle_nonce = shuffle_nonce
                if sample_randomize:
                    new_seed = self.sample_seed
                    while new_seed == self.sample_seed:
                        new_seed = int(np.random.default_rng().integers(0, 2 ** 31 - 1))
                    self.sample_seed = new_seed
                    reshuffle = True

            sample_pool_changed = False
            if reshuffle or sample_count != self.sample_count or sample_randomize != self.sample_randomize:
                sample_pool_changed = True
                self.sample_count = sample_count
                self.sample_randomize = sample_randomize
                self._load_latent_pool()
                self._rebuild_pool_dependent_state(reset_modes=False)
                # Ignore stale point/cursor data from this frame after pool reset.
                cursor = self.state.cursor.copy()
                raw_points = self.state.control_points.copy()
                decode_enabled = True
                render_enabled = False
                _debug_print_control_points(
                    "sample_pool_changed",
                    self.state.control_points,
                    extra=f"sample_count={int(self.sample_count)}, sample_randomize={bool(self.sample_randomize)}",
                )

            if dataset_name != self.dataset_name:
                self.dataset_name = self._resolve_dataset_name(dataset_name)
                self.hparams.dataset_name = self.dataset_name
                self._load_latent_pool()
                self._rebuild_pool_dependent_state(reset_modes=False)
                # Ignore stale point/cursor data from this frame after dataset swap.
                cursor = self.state.cursor.copy()
                raw_points = self.state.control_points.copy()
                decode_enabled = True
                render_enabled = False
                _debug_print_control_points(
                    "dataset_changed",
                    self.state.control_points,
                    extra=f"dataset={self.dataset_name}, sample_count={int(self.sample_count)}",
                )

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
            self.state.control_points = _clamp_points_to_extent(points, self.field_half_extent)
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

            if points.shape == prev_points_snapshot.shape and not np.allclose(points, prev_points_snapshot, atol=1e-6):
                _debug_print_control_points(
                    "frontend_points_applied",
                    points,
                    extra=f"render_enabled={bool(render_enabled)}, sample_count={int(self.sample_count)}",
                )

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

            if (not prev_render_enabled) and self.state.render_enabled and client_render_rising_edge:
                # Start a new background render with current interpolation settings.
                self.state.decode_enabled = False
                self.decode_status = "idle"
                self.decode_progress_step = -1
                self._last_triggered_latent_signature = None
                self.reconstruction_matrix.fill(0.0)
                # Reset render status immediately so the progress bar clears (shows idle)
                # before the background thread publishes its first "rendering" update.
                self.render_status = "idle"
                self.render_completed = 0
                self.render_total = 0
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
                if self.decoder_manager is not None:
                    self.decoder_manager.discard()  # decode is off: drop queued / still running results
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


def _training_device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_compile_default() -> bool:
    """torch.compile is only used on CUDA and needs a C compiler."""
    return bool(torch.cuda.is_available() and (shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")))


# Default ports: CUDA machines use a different port than MPS/CPU machines (RBF_SERVER_PORT overrides both).
PORT_CUDA = 5006
PORT_MPS_CPU = 5009


def create_app(sample_count: int = 10, sample_seed: int = 42, hparams_path: str | os.PathLike | None = None):
    controller = RBFServerController(sample_count=sample_count, sample_seed=sample_seed)
    # A running worker must never outlive the server.
    atexit.register(controller.training.terminate)
    app = Flask(__name__, template_folder=os.path.dirname(__file__))
    socketio = SocketIO(app, cors_allowed_origins="*")

    def _on_dataset_job_finished(final_state: str):
        socketio.emit("ds_list", {"datasets": dataset_tools.list_datasets(), "state": final_state})

    dataset_jobs = dataset_tools.DatasetJobManager(on_finish=_on_dataset_job_finished)
    atexit.register(dataset_jobs.cancel)
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

    def train_broadcast_loop():
        last_version = -1
        while True:
            version = controller.training.version
            if version != last_version:
                last_version = version
                socketio.emit("train_update", controller.training.snapshot())
            socketio.sleep(0.25)  # ~4 Hz

    def dataset_broadcast_loop():
        last_version = -1
        while True:
            version = dataset_jobs.version
            if version != last_version:
                last_version = version
                socketio.emit("ds_update", dataset_jobs.snapshot())
            socketio.sleep(0.25)  # ~4 Hz

    socketio.start_background_task(decode_broadcast_loop)
    socketio.start_background_task(render_broadcast_loop)
    socketio.start_background_task(train_broadcast_loop)
    socketio.start_background_task(dataset_broadcast_loop)

    @app.get("/")
    def index():
        return render_template("template.html")

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(str(assets_dir), filename)

    @app.get("/api/health")
    def health():
        status = controller._app_status()
        device = str(controller.runtime.device) if controller.runtime is not None else None
        return {"ok": True, "device": device, "cuda_ready": torch.cuda.is_available(), **status}

    @app.get("/api/train/status")
    def train_status():
        return controller.training.snapshot()

    @app.get("/api/dataset/status")
    def dataset_status():
        return dataset_jobs.snapshot()

    def run_dataset_task(event: str, func):
        """Run a (possibly slow) dataset query off the socket handler and answer only the asking client."""
        sid = request.sid

        def _work():
            try:
                payload = {"ok": True, **func()}
            except Exception as exc:
                if not isinstance(exc, (FileNotFoundError, ValueError)):
                    traceback.print_exc()
                payload = {"ok": False, "error": str(exc)}
            socketio.emit(event, payload, to=sid)

        socketio.start_background_task(_work)

    def build_dataset_state() -> dict:
        return {
            "branch": safe_branch_name(get_current_branch()),
            "data_dir": str(dataset_tools.data_dir()),
            "datasets": dataset_tools.list_datasets(),
            "job": dataset_jobs.snapshot(),
            "mappings": list(dataset_tools.MAPPING_CHOICES),
            "hparams": {
                "width": int(dataset_tools.HP.width), "n_timing": int(dataset_tools.HP.n_timing),
                "n_vel": int(dataset_tools.HP.n_vel), "max_seq_len": int(dataset_tools.HP.max_seq_len),
                "n_pitch": int(dataset_tools.HP.n_pitch), "token_format": str(dataset_tools.HP.token_format),
            },
        }

    @socketio.on("ds_get_state")
    def on_ds_get_state():
        try:
            emit("ds_state", build_dataset_state())
        except Exception as exc:
            traceback.print_exc()
            emit("error", {"message": str(exc)})

    @socketio.on("ds_browse")
    def on_ds_browse(payload=None):
        payload = payload if isinstance(payload, dict) else {}
        run_dataset_task("ds_browse_result", lambda: dataset_tools.browse_dir(
            payload.get("path"), show_hidden=bool(payload.get("show_hidden", False))))

    @socketio.on("ds_scan")
    def on_ds_scan(payload=None):
        payload = payload if isinstance(payload, dict) else {}

        def _scan():
            result = dataset_tools.scan_midi_folder(str(payload.get("path") or ""))
            if not result.get("ok"):
                raise ValueError(result.get("error", "Scan failed"))
            result.pop("ok", None)
            return result

        run_dataset_task("ds_scan_result", _scan)

    @socketio.on("ds_start")
    def on_ds_start(payload=None):
        payload = payload if isinstance(payload, dict) else {}
        kind = str(payload.get("kind") or "")
        try:
            ok, errors = dataset_jobs.start(kind, payload.get("params"))
        except Exception as exc:
            traceback.print_exc()
            ok, errors = False, {"general": str(exc)}
        emit("ds_start_result", {"ok": bool(ok), "kind": kind, "errors": errors})

    @socketio.on("ds_cancel")
    def on_ds_cancel(payload=None):
        emit("ds_cancel_result", {"ok": bool(dataset_jobs.cancel())})

    @socketio.on("ds_info")
    def on_ds_info(payload=None):
        name = str((payload or {}).get("name") or "") if isinstance(payload, dict) else ""
        run_dataset_task("ds_info_result", lambda: {"dataset": dataset_tools.dataset_info(name)})

    @socketio.on("ds_sequence")
    def on_ds_sequence(payload=None):
        payload = payload if isinstance(payload, dict) else {}
        name = str(payload.get("name") or "")
        index = _safe_int(payload.get("index"), 0)
        run_dataset_task("ds_sequence_result", lambda: {"sequence": dataset_tools.sequence_view(name, index)})

    @socketio.on("ds_save_info")
    def on_ds_save_info(payload=None):
        payload = payload if isinstance(payload, dict) else {}
        name = str(payload.get("name") or "")
        try:
            info = dataset_tools.update_dataset_info(name, payload.get("patch"))
            emit("ds_save_info_result", {"ok": True, "name": name, "info": info})
            socketio.emit("ds_list", {"datasets": dataset_tools.list_datasets(), "state": "info"})
        except Exception as exc:
            if not isinstance(exc, (FileNotFoundError, ValueError)):
                traceback.print_exc()
            emit("ds_save_info_result", {"ok": False, "name": name, "error": str(exc)})

    @socketio.on("ds_delete")
    def on_ds_delete(payload=None):
        payload = payload if isinstance(payload, dict) else {}
        name = str(payload.get("name") or "")
        try:
            if dataset_jobs.is_active():
                raise ValueError("A dataset job is running; wait until it has finished or cancel it first.")
            if controller.training.is_active():
                raise ValueError("A training is running; datasets cannot be deleted until it has ended.")
            deleted = dataset_tools.delete_dataset(name)
            with controller._lock:
                if controller.runtime is not None and controller.dataset_name == deleted:
                    # The inference pool was built from the deleted file: switch to another dataset (or idle).
                    controller._refresh_dataset(None)
                    controller._load_latent_pool()
                    controller._rebuild_pool_dependent_state(reset_modes=False)
            emit("ds_delete_result", {"ok": True, "name": deleted})
            socketio.emit("ds_list", {"datasets": dataset_tools.list_datasets(), "state": "deleted"})
        except Exception as exc:
            if not isinstance(exc, (FileNotFoundError, ValueError)):
                traceback.print_exc()
            emit("ds_delete_result", {"ok": False, "name": name, "error": str(exc)})

    def build_train_config_payload() -> dict:
        return {
            "defaults": controller.train_defaults(),
            "datasets": controller._list_available_datasets(),
            "legacy_datasets": controller._list_legacy_datasets(),
            "model_options": controller._build_model_catalog(),
            "options": {"torch_compile": _torch_compile_default(), "log_interval": 25},
            "gpus": cuda_device_list(),
            "device_plan": controller.device_plan(),
            "branch": safe_branch_name(get_current_branch()),
            "device": _training_device_label(),
            "keys": {"train": TRAIN_KEYS, "model": MODEL_KEYS, "architecture": ARCHITECTURE_KEYS},
            "state": controller.training.snapshot(),
        }

    @socketio.on("train_get_config")
    def on_train_get_config():
        try:
            emit("train_config", build_train_config_payload())
        except Exception as exc:
            traceback.print_exc()
            emit("error", {"message": str(exc)})

    @socketio.on("train_start")
    def on_train_start(payload):
        try:
            ok, errors = controller.start_training(payload)
        except Exception as exc:
            traceback.print_exc()
            ok, errors = False, {"general": str(exc)}
        emit("train_start_result", {"ok": bool(ok), "errors": errors})

    @socketio.on("train_stop")
    def on_train_stop(payload=None):
        force = bool((payload or {}).get("force", False)) if isinstance(payload, dict) else False
        emit("train_stop_result", {"ok": bool(controller.stop_training(force=force))})

    @socketio.on("hparams_save")
    def on_hparams_save(payload):
        changes = (payload or {}).get("changes", {}) if isinstance(payload, dict) else {}
        try:
            result = save_hparams(changes, path=hparams_path)
            emit("hparams_save_result", {
                "ok": True,
                "changed": result["changed"],
                "unchanged": result["unchanged"],
                "derived": result["derived"],
                "backup": result["backup"],
                "message": (
                    f"Saved {len(result['changed'])} parameter(s) to hparams.py (backup: hparams.py.bak). "
                    "Training uses the new values immediately; the running inference model is unaffected until restart."
                    if result["changed"] else "hparams.py already contains these values; nothing changed."
                ),
            })
        except HparamsSaveError as exc:
            emit("hparams_save_result", {"ok": False, "errors": exc.errors, "message": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            emit("hparams_save_result", {"ok": False, "errors": {}, "message": str(exc)})

    @socketio.on("bootstrap")
    def on_bootstrap():
        controller.recheck_resources()
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
                "decodeEnabled": bool(controller._app_status()["ready"]),
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
        # Automatic mode: CUDA machines default to PORT_CUDA, MPS and CPU machines to PORT_MPS_CPU.
        port = PORT_CUDA if torch.cuda.is_available() else PORT_MPS_CPU
        user_input_string = str(port)
        user_input_int = port
        print(f"Automatic port selection enabled. Device: {_training_device_label()}. Base port: {port}")
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
