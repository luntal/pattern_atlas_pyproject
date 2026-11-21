"""Compound token codec ("compound_v1").

A frame describes exactly one note: ``[step, pitch, velocity, deviation]`` (in this order inside the tensor).

- ``step``: absolute 16th position inside the 2-bar window (raw 0..width-1)
- ``pitch``: chromatic pitch index (raw 0..n_pitch-1)
- ``velocity``: quantised velocity (raw 0..n_vel-1)
- ``deviation``: micro timing (raw 0..n_timing-1, center = n_timing // 2)

Every field has its own index space: 0 = PAD, 1 = SOS, 2 = EOS, raw values start at ``n_specials`` (3).
A special frame carries the same id in all four fields. A sequence holds the note frames sorted by
``(step, pitch)``, then one EOS frame, then PAD frames. This module is the only place that knows the offsets.
"""
from __future__ import annotations

import numpy as np

TOKEN_FORMAT = "compound_v1"
LEGACY_FORMAT = "triplet"
N_FIELDS = 4
FIELD_STEP, FIELD_PITCH, FIELD_VEL, FIELD_DEV = 0, 1, 2, 3
FIELD_NAMES = ("step", "pitch", "velocity", "deviation")

PAD, SOS, EOS = 0, 1, 2
N_SPECIALS = 3
_SPECIAL_IDS = {"pad": PAD, "sos": SOS, "eos": EOS}

# Matrix -> frames: a cell counts as note when its peak is above this value (same as the old triplet buffer parser).
MATRIX_NOTE_THRESHOLD = 0.02


def _hp(hp=None):
    if hp is None:
        from hparams import hparams as default_hp
        return default_hp
    return hp


def field_sizes(hp=None) -> list[int]:
    """Raw value range per field ``[width, n_pitch, n_vel, n_timing]``."""
    hp = _hp(hp)
    return [int(hp.width), int(hp.n_pitch), int(hp.n_vel), int(hp.n_timing)]


def field_vocab_sizes(hp=None) -> list[int]:
    return [size + N_SPECIALS for size in field_sizes(hp)]


def center_timing(hp=None) -> int:
    return int(_hp(hp).n_timing) // 2


def note_length(hp=None) -> int:
    """Length of a note in matrix columns (half a 16th)."""
    return max(1, int(_hp(hp).n_timing) // 2)


def format_info(hp=None) -> dict:
    """Format metadata stored in datasets, checkpoints and render exports."""
    hp = _hp(hp)
    return {
        "token_format": TOKEN_FORMAT,
        "field_names": list(FIELD_NAMES),
        "field_sizes": field_sizes(hp),
        "field_vocab_sizes": field_vocab_sizes(hp),
        "n_specials": N_SPECIALS,
        "pad_token": PAD, "sos_token": SOS, "eos_token": EOS,
        "center_timing": center_timing(hp),
        "max_seq_len": int(hp.max_seq_len),
    }


def special_frame(kind) -> np.ndarray:
    """``special_frame('pad'|'sos'|'eos')`` (or the id) -> int16 [4]."""
    value = _SPECIAL_IDS[kind.lower()] if isinstance(kind, str) else int(kind)
    return np.full((N_FIELDS,), value, dtype=np.int16)


def is_compound_array(data) -> bool:
    shape = np.shape(data)
    return len(shape) == 3 and shape[-1] == N_FIELDS


def detect_format(shape) -> str:
    """Token format from an array shape: 3D with 4 fields = compound, 2D = legacy triplet."""
    shape = list(shape or [])
    if len(shape) == 3 and shape[-1] == N_FIELDS:
        return TOKEN_FORMAT
    if len(shape) == 2:
        return LEGACY_FORMAT
    return "unknown"


def _field_value(note, index: int, key: str):
    if isinstance(note, dict):
        return note[key]
    return note[index]


def _clip(value, size: int) -> int:
    return int(max(0, min(int(size) - 1, int(round(float(value))))))


def notes_to_raw(notes, hp=None) -> np.ndarray:
    """Notes (``(step, pitch, velocity, deviation)`` raw tuples or dicts) -> sorted, de-duplicated raw array [K, 4].

    Steps outside the window are dropped, the other fields are clipped to their range. Notes with the same
    ``(step, pitch)`` are merged (the loudest one wins, with its deviation).
    """
    sizes = field_sizes(hp)
    best: dict[tuple[int, int], tuple[int, int]] = {}
    for note in notes or []:
        step = int(_field_value(note, 0, "step"))
        if step < 0 or step >= sizes[FIELD_STEP]:
            continue
        pitch = _clip(_field_value(note, 1, "pitch"), sizes[FIELD_PITCH])
        vel = _clip(_field_value(note, 2, "velocity"), sizes[FIELD_VEL])
        dev = _clip(_field_value(note, 3, "deviation"), sizes[FIELD_DEV])
        key = (step, pitch)
        if key not in best or vel > best[key][0]:
            best[key] = (vel, dev)
    if not best:
        return np.zeros((0, N_FIELDS), dtype=np.int16)
    rows = [(s, p, v, d) for (s, p), (v, d) in sorted(best.items())]
    return np.asarray(rows, dtype=np.int16)


def notes_to_frames(notes, hp=None, max_len: int | None = None, return_info: bool = False):
    """Notes -> int16 [max_len, 4] frames (sorted, merged duplicates, EOS, PAD).

    Too many notes are cut after time (first ``max_len - 1`` notes plus EOS). With ``return_info`` the result is
    ``(frames, n_notes, truncated)``.
    """
    hp = _hp(hp)
    max_len = int(max_len or hp.max_seq_len)
    raw = notes_to_raw(notes, hp)
    truncated = raw.shape[0] > max_len - 1
    raw = raw[: max_len - 1]
    frames = np.zeros((max_len, N_FIELDS), dtype=np.int16)
    n = int(raw.shape[0])
    frames[:n] = raw + N_SPECIALS
    frames[n] = EOS
    if return_info:
        return frames, n, bool(truncated)
    return frames


def valid_note_mask(frames, hp=None) -> np.ndarray:
    """Bool mask [..., T]: frames that are real notes (all fields hold values, before the first EOS frame)."""
    f = np.asarray(frames)
    vocab = np.asarray(field_vocab_sizes(hp))
    is_value = np.all((f >= N_SPECIALS) & (f < vocab), axis=-1)
    ended = np.cumsum(f[..., FIELD_STEP] == EOS, axis=-1) > 0
    return is_value & ~ended


def frames_to_raw(frames, hp=None) -> np.ndarray:
    """Frames -> raw note values [K, 4] (only the notes before the first EOS)."""
    f = np.asarray(frames).reshape(-1, N_FIELDS)
    mask = valid_note_mask(f, hp)
    return (f[mask].astype(np.int64) - N_SPECIALS).astype(np.int16)


def frames_to_notes(frames, hp=None) -> list[dict]:
    """Frames -> note list for viewer and playback (same structure as the former triplet parser)."""
    hp = _hp(hp)
    half = note_length(hp)
    center = center_timing(hp)
    n_vel = int(hp.n_vel)
    notes = []
    for step, pitch, vel, dev in frames_to_raw(frames, hp).tolist():
        shift = (dev - center) / half
        vel_norm = vel / max(1, n_vel - 1)
        notes.append({
            "step": int(step),
            "pos": round(step + 0.5 * shift, 4),
            "shift": round(shift, 4),
            "pitch": int(pitch),
            "midi": int(pitch + 36),
            "vel": int(round(vel_norm * 127)),
            "vel_norm": round(float(vel_norm), 4),
            "velocity": int(vel),
            "deviation": int(dev),
        })
    return notes


def empty_matrix(hp=None) -> np.ndarray:
    hp = _hp(hp)
    return np.zeros((int(hp.height), int(hp.width) * int(hp.n_timing)), dtype=np.float32)


def write_wrapped_segment(matrix: np.ndarray, row: int, start: int, length: int, value: float):
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


def write_raw_note(matrix: np.ndarray, step: int, pitch: int, vel: int, dev: int, hp=None) -> int:
    """Write one raw note into the matrix. Returns the frontier column (end of the note's 16th)."""
    hp = _hp(hp)
    n_timing = int(hp.n_timing)
    note_len = note_length(hp)
    start = int(step) * n_timing + (int(dev) - center_timing(hp))
    value = float(vel) / max(1, int(hp.n_vel) - 1)
    if 0 <= int(pitch) < matrix.shape[0]:
        write_wrapped_segment(matrix, int(pitch), start, note_len, value)
    return max(0, min(matrix.shape[1], int(step) * n_timing + note_len))


def write_frame(matrix: np.ndarray, frame, hp=None) -> int | None:
    """Write one (indexed) frame into the matrix; returns the frontier column or None for non-note frames."""
    f = np.asarray(frame).reshape(N_FIELDS)
    if not bool(valid_note_mask(f.reshape(1, N_FIELDS), hp)[0]):
        return None
    step, pitch, vel, dev = (int(x) - N_SPECIALS for x in f.tolist())
    return write_raw_note(matrix, step, pitch, vel, dev, hp)


def frames_to_matrix(frames, hp=None) -> np.ndarray:
    """Frames -> float32 [height, width * n_timing]. Column = step * n_timing + (deviation - center)."""
    hp = _hp(hp)
    matrix = empty_matrix(hp)
    for step, pitch, vel, dev in frames_to_raw(frames, hp).tolist():
        write_raw_note(matrix, step, pitch, vel, dev, hp)
    return matrix


def matrix_to_notes(matrix, hp=None) -> list[tuple[int, int, int, int]]:
    """Note matrix -> raw notes ``(step, pitch, velocity, deviation)`` (exact inverse of ``frames_to_matrix``).

    A note starts where a row becomes active (value above the threshold) or changes its value. The onset
    column ``c = step * n_timing + deviation - center`` (wrapped) gives ``step`` and ``deviation`` back.
    """
    hp = _hp(hp)
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError("matrix must be a 2D list.")
    if m.shape[0] != int(hp.height):
        raise ValueError(f"matrix row count must be {int(hp.height)}.")
    n_timing = int(hp.n_timing)
    width = int(hp.width)
    target_cols = width * n_timing
    if m.shape[1] < target_cols:
        m = np.pad(m, ((0, 0), (0, target_cols - m.shape[1])))
    elif m.shape[1] > target_cols:
        m = m[:, :target_cols]
    n_vel = int(hp.n_vel)
    center = center_timing(hp)
    active = m > MATRIX_NOTE_THRESHOLD
    prev = np.roll(m, 1, axis=1)
    prev_active = np.roll(active, 1, axis=1)
    onsets = active & (~prev_active | (m != prev))
    notes = []
    for pitch, col in zip(*np.nonzero(onsets)):
        shifted = (int(col) + center) % target_cols
        step, dev = divmod(shifted, n_timing)
        vel = max(0, min(n_vel - 1, int(round(float(m[pitch, col]) * max(1, n_vel - 1)))))
        notes.append((int(step), int(pitch), vel, int(dev)))
    return notes


def matrix_to_frames(matrix, hp=None, max_len: int | None = None) -> np.ndarray:
    """Note matrix -> int16 frames (sorted by (step, pitch), duplicates merged, EOS at the end).

    Without ``max_len`` the result is ``[K + 1, 4]`` (notes + EOS); with ``max_len`` it is cut / padded to it.
    """
    hp = _hp(hp)
    notes = matrix_to_notes(matrix, hp)
    if max_len is not None:
        return notes_to_frames(notes, hp, max_len=max_len)
    raw = notes_to_raw(notes, hp)
    frames = np.zeros((raw.shape[0] + 1, N_FIELDS), dtype=np.int16)
    frames[:raw.shape[0]] = raw + N_SPECIALS
    frames[raw.shape[0]] = EOS
    return frames


def fit_frames(frames, max_len: int) -> np.ndarray:
    """One frame sequence -> exactly ``max_len`` frames (cut with EOS as last frame, or PAD filled)."""
    f = np.asarray(frames, dtype=np.int64).reshape(-1, N_FIELDS)
    max_len = int(max_len)
    out = np.zeros((max_len, N_FIELDS), dtype=np.int16)
    if f.shape[0] > max_len:
        f = f[:max_len].copy()
        if not np.any(f[:, FIELD_STEP] == EOS):
            f[-1] = EOS
    out[:f.shape[0]] = f
    return out


def pad_frames(seqs, max_len: int) -> np.ndarray:
    """List of frame sequences -> int16 [N, max_len, 4]."""
    seqs = list(seqs)
    out = np.zeros((len(seqs), int(max_len), N_FIELDS), dtype=np.int16)
    for i, seq in enumerate(seqs):
        out[i] = fit_frames(seq, max_len)
    return out


def resize_frames_array(data, max_len: int):
    """Frame array [N, T, 4] -> [N, max_len, 4]: PAD frames are appended, or too long sequences are cut after
    time (the last frame becomes EOS when no EOS is left). Returns (array, n_cut_sequences)."""
    data = np.asarray(data)
    max_len = int(max_len)
    n, t, _ = data.shape
    if t == max_len:
        return data, 0
    if t < max_len:
        out = np.zeros((n, max_len, N_FIELDS), dtype=data.dtype)
        out[:, :t] = data
        return out, 0
    out = data[:, :max_len].copy()
    lost = (data[:, max_len:, FIELD_STEP] != PAD).any(axis=1)  # real frames (incl. EOS) behind the cut
    needs_eos = lost & ~(out[:, :, FIELD_STEP] == EOS).any(axis=1)
    out[needs_eos, max_len - 1] = EOS
    return out, int(lost.sum())


def sequence_length(frames) -> int:
    """Number of frames up to and including the first EOS (or all non-PAD frames)."""
    f = np.asarray(frames).reshape(-1, N_FIELDS)
    eos = np.flatnonzero(f[:, FIELD_STEP] == EOS)
    if eos.size:
        return int(eos[0]) + 1
    return int(np.sum(f[:, FIELD_STEP] != PAD))
