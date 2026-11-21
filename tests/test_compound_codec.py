"""Checks the compound token codec (src/modules/compound.py).

Run from the project root:  python tests/test_compound_codec.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hparams import hparams as HP  # noqa: E402
from modules import compound  # noqa: E402

# Layout of the former flat triplet vocabulary (only used as reference here).
OLD_EOS, OLD_STEP = 2, 3
OLD_MIN_PITCH, OLD_MIN_VEL, OLD_MIN_TIMING = 4, 58, 90
OLD_MAX_PITCH, OLD_MAX_VEL, OLD_MAX_TIMING = 58, 90, 123
OLD_CENTER_TIMING = OLD_MIN_TIMING + 33 // 2


def old_triplet_tokens(notes):
    """Raw notes (step, pitch, vel, dev) -> former flat triplet sequence (pitch, vel, timing + STEP tokens)."""
    tokens, last = [], 0
    for step, pitch, vel, dev in sorted(notes):
        tokens.extend([OLD_STEP] * (step - last))
        last = step
        tokens.extend([OLD_MIN_PITCH + pitch, OLD_MIN_VEL + vel, OLD_MIN_TIMING + dev])
    tokens.extend([OLD_STEP] * (int(HP.width) - last))
    tokens.append(OLD_EOS)
    return tokens


def old_tokens_to_matrix(tokens):
    """Copy of the former `CudaRBFRuntime._tokens_to_matrix` triplet parser."""
    n_timing = int(HP.n_timing)
    matrix = np.zeros((int(HP.height), int(HP.width) * n_timing), dtype=np.float32)
    note_len = max(1, n_timing // 2)
    step_idx, pending_pitch, pending_vel = 0, None, None
    for token in tokens:
        if token == OLD_EOS:
            break
        if token == OLD_STEP:
            step_idx += 1
            continue
        if OLD_MIN_PITCH <= token < OLD_MAX_PITCH:
            pending_pitch, pending_vel = token, None
            continue
        if pending_pitch is not None and OLD_MIN_VEL <= token < OLD_MAX_VEL:
            pending_vel = token
            continue
        if pending_pitch is not None and pending_vel is not None and OLD_MIN_TIMING <= token < OLD_MAX_TIMING:
            vel = (pending_vel - OLD_MIN_VEL) / max(1, int(HP.n_vel) - 1)
            start = step_idx * n_timing + (token - OLD_CENTER_TIMING)
            compound.write_wrapped_segment(matrix, pending_pitch - OLD_MIN_PITCH, start, note_len, float(vel))
            pending_pitch = pending_vel = None
    return matrix


def random_notes(rng, n):
    """Unique (step, pitch) notes with random velocity / deviation."""
    keys = rng.sample([(s, p) for s in range(int(HP.width)) for p in range(int(HP.n_pitch))], n)
    return [(s, p, rng.randrange(int(HP.n_vel)), rng.randrange(int(HP.n_timing))) for s, p in keys]


def check_format():
    assert HP.token_format == compound.TOKEN_FORMAT
    assert list(HP.field_vocab_sizes) == [35, 57, 35, 36] == compound.field_vocab_sizes(HP)
    assert (HP.pad_token, HP.sos_token, HP.eos_token) == (compound.PAD, compound.SOS, compound.EOS)
    assert HP.center_timing == 16 and int(HP.max_seq_len) == 128
    assert compound.special_frame("eos").tolist() == [2, 2, 2, 2]
    assert compound.detect_format((10, 128, 4)) == "compound_v1"
    assert compound.detect_format((10, 256)) == "triplet"


def check_roundtrip_and_order():
    rng = random.Random(1)
    for _ in range(50):
        notes = random_notes(rng, rng.randrange(0, 60))
        frames, n, truncated = compound.notes_to_frames(rng.sample(notes, len(notes)), HP, return_info=True)
        assert frames.shape == (int(HP.max_seq_len), 4) and frames.dtype == np.int16
        assert n == len(notes) and not truncated
        assert frames[n].tolist() == [2, 2, 2, 2], "EOS frame expected after the notes"
        assert not frames[n + 1:].any(), "PAD after EOS expected"
        raw = compound.frames_to_raw(frames, HP).tolist()
        assert raw == [list(n) for n in sorted(notes)], "lossless roundtrip / sort order by (step, pitch)"
        keys = [(r[0], r[1]) for r in raw]
        assert keys == sorted(keys)
        notes_view = compound.frames_to_notes(frames, HP)
        assert len(notes_view) == n and all({"step", "pos", "shift", "pitch", "midi", "vel", "vel_norm"} <= set(x) for x in notes_view)


def check_duplicates_and_clipping():
    notes = [(4, 10, 5, 16), (4, 10, 20, 3), (4, 10, 7, 30), (40, 1, 1, 1), (0, 99, 99, -5)]
    raw = compound.frames_to_raw(compound.notes_to_frames(notes, HP), HP).tolist()
    assert raw == [[0, 53, 31, 0], [4, 10, 20, 3]], raw  # loudest duplicate wins, step 40 dropped, others clipped


def check_truncation_and_empty():
    notes = [(s, p, 10, 16) for s in range(32) for p in range(8)]  # 256 notes
    frames, n, truncated = compound.notes_to_frames(notes, HP, max_len=128, return_info=True)
    assert truncated and n == 127 and frames[-1].tolist() == [2, 2, 2, 2]
    raw = compound.frames_to_raw(frames, HP)
    assert raw[-1, 0] == 15, "cut after time (first notes are kept)"
    empty = compound.notes_to_frames([], HP)
    assert empty[0].tolist() == [2, 2, 2, 2] and not empty[1:].any()
    assert compound.frames_to_notes(empty, HP) == [] and not compound.frames_to_matrix(empty, HP).any()
    assert compound.sequence_length(empty) == 1
    padded = compound.pad_frames([frames[:10], empty[:1]], 6)
    assert padded.shape == (2, 6, 4) and padded[0, -1].tolist() == [2, 2, 2, 2] and padded[1, 0].tolist() == [2, 2, 2, 2]


def check_matrix_equals_old_parser():
    rng = random.Random(2)
    for _ in range(50):
        notes = random_notes(rng, rng.randrange(0, 50))
        new = compound.frames_to_matrix(compound.notes_to_frames(notes, HP), HP)
        old = old_tokens_to_matrix(old_triplet_tokens(notes))
        assert np.array_equal(new, old), "frames_to_matrix differs from the former triplet parser"


def check_matrix_roundtrip():
    rng = random.Random(3)
    width = int(HP.width)
    for _ in range(60):
        # Velocity >= 1 so the note is visible in the matrix. Same-pitch notes in consecutive steps whose
        # segments overlap (early note late, next note early) cannot be separated and are left out.
        candidates = {(s, p): (max(1, v), d) for s, p, v, d in random_notes(rng, rng.randrange(0, 60))}
        notes = []
        for (s, p), (v, d) in candidates.items():
            nxt = candidates.get(((s + 1) % width, p))
            prv = candidates.get(((s - 1) % width, p))
            if (nxt and d - nxt[1] > 16) or (prv and prv[1] - d > 16):
                continue
            notes.append((s, p, v, d))
        notes.append((0, 5, 20, 2))  # starts in the last 16th of the window (wraps around)
        notes = [n for n in notes if not (n[1] == 5 and n[0] in (1, 31) and n != (0, 5, 20, 2))]
        notes = list({(n[0], n[1]): n for n in notes}.values())
        frames = compound.notes_to_frames(notes, HP)
        matrix = compound.frames_to_matrix(frames, HP)
        back = compound.matrix_to_frames(matrix, HP)
        assert back[-1].tolist() == [2, 2, 2, 2]
        assert compound.frames_to_raw(back, HP).tolist() == [list(n) for n in sorted(notes)], "matrix -> frames roundtrip"
        fixed = compound.matrix_to_frames(matrix, HP, max_len=int(HP.max_seq_len))
        assert np.array_equal(fixed, frames)


def check_valid_mask():
    frames = np.array([[3, 3, 3, 3], [5, 1, 4, 4], [2, 2, 2, 2], [6, 6, 6, 6], [0, 0, 0, 0]], dtype=np.int16)
    assert compound.valid_note_mask(frames, HP).tolist() == [True, False, False, False, False]


if __name__ == '__main__':
    check_format()
    check_roundtrip_and_order()
    check_duplicates_and_clipping()
    check_truncation_and_empty()
    check_matrix_equals_old_parser()
    check_matrix_roundtrip()
    check_valid_mask()
    print("ALL OK: compound codec")
