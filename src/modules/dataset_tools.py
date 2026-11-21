"""Server-side dataset tools used by the "Dataset" tab of rbf_server_app.

Everything that touches files or does real work lives here; the browser is only GUI and visualization.

- Server folder browser (``browse_dir`` / ``scan_midi_folder``)
- Dataset listing, statistics and per-sequence note lists (``list_datasets``, ``dataset_info``, ``sequence_view``)
- Dataset meta info: a JSON string stored under the key ``info`` inside the ``.npz`` next to ``data`` and
  ``metadata`` (``update_dataset_info``). Older datasets without the key keep working.
- ``DatasetJobManager``: one background job at a time (MIDI extraction = data.py, merge = data_merge.py,
  augmentation = data_augment.py) with progress, logs and cancel.

Datasets are always written to ``<project_root>/data/<git-branch>/<name>.npz``.

Token format: ``data`` holds compound frames ``[N, max_seq_len, 4]`` (int16, see ``modules.compound``) and
``info.token_format == "compound_v1"``. Older 2D triplet datasets are listed as ``legacy (triplet)`` and are
blocked for training, inference, merge and augmentation.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
import traceback
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path

# Keep matplotlib (imported by midi_processor) away from GUI backends inside server threads.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from hparams import hparams as HP
from modules import compound
from modules.branch_paths import get_current_branch, project_root, safe_branch_name
from modules.midi_mappings import chromatic_to_generic

INFO_KEY = "info"
INFO_SCHEMA = 1
MAX_BROWSE_ENTRIES = 3000
MAX_LIST_FILES = 200
USER_INFO_FIELDS = ("description", "notes", "tags", "custom")
MAPPING_CHOICES = ("groove_to_specific", "generic_midi_mapping", "generic_to_chromatic", "none")

_NUMBERED_DUPLICATE = re.compile(r"\.\d+\.mid[i]?$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


# ---------------------------------------------------------------------------------------------- paths
def data_dir(create: bool = False) -> Path:
    path = project_root() / "data" / safe_branch_name(get_current_branch())
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def validate_dataset_name(raw) -> tuple[str | None, str | None]:
    """Return (normalized file name with .npz, error)."""
    name = str(raw or "").strip()
    if not name:
        return None, "Enter a dataset name."
    if name.lower().endswith(".npz"):
        name = name[:-4]
    if not name or len(name) > 120 or not _NAME_RE.match(name) or name.endswith((".", " ")):
        return None, "Use letters, digits, space, '.', '_' or '-' (max. 120 characters); no path separators."
    return name + ".npz", None


def existing_dataset_path(raw) -> tuple[Path | None, str | None]:
    """Resolve an existing dataset of the current branch by plain file name."""
    name = str(raw or "").strip()
    if not name or os.path.basename(name) != name or name in (".", ".."):
        return None, "Choose a dataset from the list."
    path = data_dir() / name
    if not path.is_file():
        return None, f"Dataset '{name}' not found in {data_dir()}"
    return path, None


# ---------------------------------------------------------------------------------------------- folder browser
def _is_midi_name(name: str) -> bool:
    return ".mid" in name.lower()


def _shortcuts() -> list[dict]:
    items = [{"label": "Home", "path": str(Path.home())}]
    midifiles = project_root() / "midifiles"
    if midifiles.is_dir():
        items.append({"label": "midifiles (project)", "path": str(midifiles)})
    items.append({"label": "Project root", "path": str(project_root())})
    volumes = Path("/Volumes")
    if volumes.is_dir():
        items.append({"label": "Volumes", "path": str(volumes)})
    items.append({"label": "/", "path": os.path.abspath(os.sep)})
    return items


def browse_dir(path: str | None = None, show_hidden: bool = False) -> dict:
    """List sub folders and MIDI files of a folder on the server."""
    start = str(path or "").strip()
    if start:
        candidate = Path(start).expanduser()
    else:
        midifiles = project_root() / "midifiles"
        candidate = midifiles if midifiles.is_dir() else Path.home()
    try:
        candidate = candidate.resolve()
    except (OSError, RuntimeError):
        candidate = Path.home()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.is_file():
        candidate = candidate.parent

    dirs: list[dict] = []
    midi_files: list[str] = []
    error = None
    truncated = False
    try:
        with os.scandir(candidate) as entries:
            for count, entry in enumerate(entries):
                if count >= MAX_BROWSE_ENTRIES:
                    truncated = True
                    break
                name = entry.name
                if not show_hidden and name.startswith("."):
                    continue
                try:
                    if entry.is_dir():
                        dirs.append({"name": name, "path": str(Path(candidate) / name)})
                    elif entry.is_file() and _is_midi_name(name):
                        midi_files.append(name)
                except OSError:
                    continue
    except PermissionError:
        error = "Permission denied"
    except OSError as exc:
        error = str(exc)

    dirs.sort(key=lambda d: d["name"].lower())
    midi_files.sort(key=str.lower)
    parent = candidate.parent
    return {
        "path": str(candidate),
        "name": candidate.name or str(candidate),
        "parent": str(parent) if parent != candidate else None,
        "dirs": dirs,
        "midi_count": len(midi_files),
        "midi_files": midi_files[:MAX_LIST_FILES],
        "truncated": truncated,
        "error": error,
        "shortcuts": _shortcuts(),
    }


def scan_midi_folder(path: str, max_seconds: float = 15.0) -> dict:
    """Count MIDI files below a server folder the same way MidiProcessor.extract selects them."""
    root = Path(str(path or "")).expanduser()
    if not root.is_dir():
        return {"path": str(root), "ok": False, "error": "Folder does not exist on the server."}
    started = time.time()
    midi_files = skipped_numbered = folders = 0
    truncated = False
    for _, dirnames, filenames in os.walk(root):
        folders += 1
        for fname in filenames:
            if not _is_midi_name(fname):
                continue
            if _NUMBERED_DUPLICATE.search(fname):
                skipped_numbered += 1
            else:
                midi_files += 1
        if time.time() - started > max_seconds:
            truncated = True
            break
    return {
        "path": str(root), "ok": True, "midi_files": midi_files, "skipped_numbered_duplicates": skipped_numbered,
        "folders": folders, "truncated": truncated,
    }


# ---------------------------------------------------------------------------------------------- meta helpers
def _meta_entry_list(entry) -> list:
    if isinstance(entry, np.ndarray):
        entry = entry.tolist()
    if isinstance(entry, (list, tuple)):
        return list(entry)
    if entry is None:
        return []
    return [entry]


def _meta_name(entry, fallback: str = "") -> str:
    items = _meta_entry_list(entry)
    return str(items[0]) if items else fallback


def _meta_window(entry):
    items = _meta_entry_list(entry)
    if len(items) > 1:
        try:
            return int(items[1])
        except (TypeError, ValueError):
            return None
    return None


def _meta_bpm(entry):
    items = _meta_entry_list(entry)
    if len(items) > 2:
        try:
            bpm = float(items[2])
            if bpm > 0 and np.isfinite(bpm):
                return bpm
        except (TypeError, ValueError):
            pass
    return None


def _normalize_meta_entry(raw, idx: int) -> tuple:
    """(filename, window[, bpm]) – keeps the BPM when present."""
    items = _meta_entry_list(raw)
    name = str(items[0]) if items else str(idx)
    try:
        window = int(items[1]) if len(items) > 1 else 0
    except (TypeError, ValueError):
        window = 0
    if len(items) > 2:
        return (name, window) + tuple(items[2:])
    return (name, window)


def _metadata_array(entries: list) -> np.ndarray:
    return np.array(entries, dtype=object)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean_user_patch(patch) -> tuple[dict, dict]:
    """Validate the user editable part of the dataset info. Returns (clean, errors)."""
    clean: dict = {}
    errors: dict = {}
    if not isinstance(patch, dict):
        return clean, {"general": "Invalid meta info."}
    if "description" in patch:
        clean["description"] = str(patch.get("description") or "")[:2000]
    if "notes" in patch:
        clean["notes"] = str(patch.get("notes") or "")[:20000]
    if "tags" in patch:
        tags = patch.get("tags")
        if isinstance(tags, str):
            tags = [t for t in re.split(r"[,\n]", tags)]
        if not isinstance(tags, (list, tuple)):
            errors["tags"] = "Tags must be a list."
        else:
            seen: list[str] = []
            for tag in tags:
                tag = str(tag).strip()[:40]
                if tag and tag not in seen:
                    seen.append(tag)
            clean["tags"] = seen[:50]
    if "custom" in patch:
        custom = patch.get("custom")
        if not isinstance(custom, dict):
            errors["custom"] = "Custom fields must be key/value pairs."
        else:
            out = {}
            for key, value in custom.items():
                key = str(key).strip()[:60]
                if key:
                    out[key] = str(value if value is not None else "")[:500]
                if len(out) >= 50:
                    break
            clean["custom"] = out
    return clean, errors


def _base_info(name: str, kind: str) -> dict:
    return {
        "schema": INFO_SCHEMA, "name": name, "kind": kind, "branch": safe_branch_name(get_current_branch()),
        "created": _now_iso(), "modified": _now_iso(), "source": {}, "options": {}, "stats": {},
        "description": "", "notes": "", "tags": [], "custom": {},
    }


def _info_from_json(raw) -> dict:
    try:
        info = json.loads(str(raw))
        return info if isinstance(info, dict) else {}
    except (TypeError, ValueError):
        return {}


def _complete_info(info: dict, name: str, path: Path | None = None) -> dict:
    """Fill defaults so the GUI can rely on all keys (also for legacy datasets without info)."""
    full = _base_info(name, "legacy")
    full["created"] = None
    full["modified"] = None
    if path is not None and path.exists():
        full["modified"] = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    full.update({k: v for k, v in (info or {}).items() if v is not None})
    full["has_info"] = bool(info)
    return full


# ---------------------------------------------------------------------------------------------- token format
LEGACY_MESSAGE = "legacy triplet dataset, re-extract it in the Dataset tab"


def dataset_format(shape, info: dict | None = None) -> str:
    """Token format of a dataset: 'compound_v1', 'triplet' (legacy) or 'unknown'."""
    detected = compound.detect_format(shape)
    declared = str((info or {}).get("token_format") or "")
    if detected == compound.TOKEN_FORMAT and declared not in ("", compound.TOKEN_FORMAT):
        return "unknown"
    return detected


def is_compatible(shape, info: dict | None = None) -> bool:
    """Only compound datasets can be used for training, inference, merge and augmentation."""
    return dataset_format(shape, info) == compound.TOKEN_FORMAT


def dataset_format_of(name: str) -> tuple[str, list | None]:
    """(token_format, shape) of an existing dataset of the current branch (cheap header read)."""
    path, error = existing_dataset_path(name)
    if error:
        return "missing", None
    header = _read_npz_header(path)
    return dataset_format(header["shape"], header["info"]), header["shape"]


# ---------------------------------------------------------------------------------------------- read / write
def _read_npz_header(path: Path) -> dict:
    """Cheap listing information (shape, info) without loading the whole array."""
    result = {"shape": None, "info": {}}
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for key in ("data", "tensor"):
                member = f"{key}.npy"
                if member in names:
                    with zf.open(member) as fh:
                        version = np.lib.format.read_magic(fh)
                        if version == (1, 0):
                            shape, _, _ = np.lib.format.read_array_header_1_0(fh)
                        else:
                            shape, _, _ = np.lib.format.read_array_header_2_0(fh)
                    result["shape"] = [int(s) for s in shape]
                    break
            member = f"{INFO_KEY}.npy"
            if member in names:
                arr = np.load(io.BytesIO(zf.read(member)), allow_pickle=False)
                result["info"] = _info_from_json(arr[()])
    except Exception:
        pass
    return result


def list_datasets() -> list[dict]:
    folder = data_dir()
    if not folder.is_dir():
        return []
    items = []
    for path in sorted(folder.glob("*.npz"), key=lambda p: p.name.lower()):
        try:
            stat = path.stat()
        except OSError:
            continue
        header = _read_npz_header(path)
        info = header["info"]
        shape = header["shape"]
        token_format = dataset_format(shape, info)
        compatible = token_format == compound.TOKEN_FORMAT
        items.append({
            "name": path.name,
            "size": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "n_sequences": shape[0] if shape else None,
            "max_len": shape[1] if shape and len(shape) > 1 else None,
            "kind": info.get("kind", "legacy") if compatible else "legacy (triplet)" if token_format == compound.LEGACY_FORMAT else "unknown",
            "description": str(info.get("description", ""))[:120],
            "token_format": token_format,
            "compatible": compatible,
        })
    return items


_load_lock = threading.Lock()
_load_cache: dict = {"key": None, "value": None}
_stats_cache: dict = {}


def _file_key(path: Path):
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def load_dataset_file(name: str) -> dict:
    """Load data + metadata + info of a dataset of the current branch (the last one is cached)."""
    path, error = existing_dataset_path(name)
    if error:
        raise FileNotFoundError(error)
    key = _file_key(path)
    with _load_lock:
        if _load_cache["key"] == key:
            return _load_cache["value"]
    npz = np.load(path, allow_pickle=True)
    if "data" in npz.files:
        data = npz["data"]
    elif "tensor" in npz.files:
        data = npz["tensor"]
    else:
        raise ValueError(f"Dataset {name} contains neither 'data' nor 'tensor'.")
    metadata = npz["metadata"].tolist() if "metadata" in npz.files else []
    info = _info_from_json(npz[INFO_KEY][()]) if INFO_KEY in npz.files else {}
    value = {"name": path.name, "path": path, "data": np.asarray(data), "metadata": metadata, "info": info, "key": key}
    with _load_lock:
        _load_cache["key"] = key
        _load_cache["value"] = value
    return value


def _invalidate_cache():
    with _load_lock:
        _load_cache["key"] = None
        _load_cache["value"] = None
        _stats_cache.clear()


def pad_frames(sequences: list, max_len: int | None = None) -> np.ndarray:
    """Frame sequences -> int16 [N, max_len, 4] (cut with EOS as last frame, PAD filled)."""
    return compound.pad_frames(sequences, int(max_len or HP.max_seq_len))


def save_dataset(name: str, data: np.ndarray, metadata: list, info: dict, overwrite: bool = False) -> Path:
    """Write ``data``, ``metadata`` and ``info`` atomically to data/<branch>/<name>."""
    folder = data_dir(create=True)
    path = folder / name
    if path.exists() and not overwrite:
        raise FileExistsError(f"Dataset '{name}' already exists in {folder}")
    data = np.asarray(data)
    if not compound.is_compound_array(data):
        raise ValueError(f"Only compound frame datasets [N, T, 4] can be saved (got shape {list(data.shape)}).")
    data = data.astype(np.int16)
    info = dict(info)
    info.update(compound.format_info(HP))
    info["max_seq_len"] = int(data.shape[1])
    info["name"] = name
    info["modified"] = _now_iso()
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, data=data, metadata=_metadata_array(metadata), **{INFO_KEY: np.array(json.dumps(info))})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    _invalidate_cache()
    return path


def delete_dataset(name: str) -> str:
    """Delete a dataset file of the current branch (plain file name only). Returns the deleted file name."""
    path, error = existing_dataset_path(name)
    if error:
        raise FileNotFoundError(error)
    path.unlink()
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    _invalidate_cache()
    return path.name


def update_dataset_info(name: str, patch) -> dict:
    """Update the user editable meta info of an existing dataset (rewrites the file atomically)."""
    clean, errors = _clean_user_patch(patch)
    if errors:
        raise ValueError("; ".join(f"{k}: {v}" for k, v in errors.items()))
    path, error = existing_dataset_path(name)
    if error:
        raise FileNotFoundError(error)
    npz = np.load(path, allow_pickle=True)
    arrays = {key: npz[key] for key in npz.files if key != INFO_KEY}
    info = _info_from_json(npz[INFO_KEY][()]) if INFO_KEY in npz.files else {}
    base = _base_info(path.name, "legacy")
    base["created"] = None
    base.update(info)
    base.update(clean)
    base["name"] = path.name
    base["modified"] = _now_iso()
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays, **{INFO_KEY: np.array(json.dumps(base))})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    _invalidate_cache()
    return _complete_info(base, path.name, path)


# ---------------------------------------------------------------------------------------------- statistics
def pitch_labels() -> list[str]:
    """Chromatic pitch index -> label (from the compressed generic mapping CSV, '' if unknown)."""
    n_pitch = int(HP.n_pitch)
    labels = [""] * n_pitch
    csv_path = project_root() / "static" / "generic_midi_mapping_compressed.csv"
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                chromatic = (row.get("Chromatic MIDI Pitch") or "").strip()
                sound = (row.get("Sound") or "").strip()
                if chromatic and sound and 0 <= int(chromatic) < n_pitch:
                    labels[int(chromatic)] = sound
    except Exception:
        pass
    return labels


def compute_stats(data: np.ndarray, metadata: list) -> dict:
    d = np.asarray(data)
    token_format = dataset_format(d.shape)
    if token_format != compound.TOKEN_FORMAT or d.shape[0] == 0:
        return {
            "n_sequences": int(d.shape[0]) if d.ndim else 0,
            "max_len": int(d.shape[1]) if d.ndim > 1 else 0,
            "token_format": token_format,
            "legacy": token_format != compound.TOKEN_FORMAT,
        }
    n, length = int(d.shape[0]), int(d.shape[1])
    n_specials = compound.N_SPECIALS
    note_mask = compound.valid_note_mask(d, HP)  # [N, T]
    notes_per = note_mask.sum(axis=1)
    lengths = (d[..., compound.FIELD_STEP] != compound.PAD).sum(axis=1)
    has_eos = (d[..., compound.FIELD_STEP] == compound.EOS).any(axis=1)
    raw = d[note_mask].astype(np.int64) - n_specials  # [K, 4]
    step_hits = np.zeros((n, int(HP.width)), dtype=bool)
    seq_idx = np.nonzero(note_mask)[0]
    step_hits[seq_idx, raw[:, compound.FIELD_STEP]] = True
    steps_per = step_hits.sum(axis=1)
    truncated = int((notes_per >= length - 1).sum())

    n_pitch = int(HP.n_pitch)
    pitch_hist = np.bincount(raw[:, compound.FIELD_PITCH], minlength=n_pitch)[:n_pitch]
    vel_hist = np.bincount(raw[:, compound.FIELD_VEL], minlength=int(HP.n_vel))[:int(HP.n_vel)]
    timing_hist = np.bincount(raw[:, compound.FIELD_DEV], minlength=int(HP.n_timing))[:int(HP.n_timing)]
    step_hist = np.bincount(raw[:, compound.FIELD_STEP], minlength=int(HP.width))[:int(HP.width)]
    max_bins = 64
    notes_hist = np.bincount(np.minimum(notes_per, max_bins - 1).astype(int))

    rows = np.ascontiguousarray(d).reshape(n, -1)
    unique_rows = len({rows[i].tobytes() for i in range(n)})
    names = {_meta_name(entry).split(" | ")[0] for entry in metadata} if metadata else set()
    bpms = np.array([b for b in (_meta_bpm(entry) for entry in metadata) if b is not None], dtype=float)

    bpm_info = {"count": int(bpms.size), "min": None, "max": None, "mean": None, "median": None, "edges": [], "counts": []}
    if bpms.size:
        lo, hi = float(bpms.min()), float(bpms.max())
        bins = 20 if hi > lo else 1
        counts, edges = np.histogram(bpms, bins=bins, range=(lo, hi if hi > lo else lo + 1))
        bpm_info.update({
            "min": round(lo, 2), "max": round(hi, 2), "mean": round(float(bpms.mean()), 2),
            "median": round(float(np.median(bpms)), 2),
            "edges": [round(float(e), 2) for e in edges], "counts": [int(c) for c in counts],
        })

    frames_info = {
        "mean": round(float(lengths.mean()), 2), "min": int(lengths.min()), "max": int(lengths.max()),
        "total": int(lengths.sum()),
    }
    return {
        "token_format": token_format,
        "legacy": False,
        "n_sequences": int(n),
        "max_len": int(length),
        "unique_sequences": int(unique_rows),
        "duplicate_sequences": int(n - unique_rows),
        "without_eos": int((~has_eos).sum()),
        "empty_sequences": int((notes_per == 0).sum()),
        "truncated_sequences": truncated,
        "max_notes": int(notes_per.max()),
        # frames per sequence including EOS (key kept as "tokens" for the GUI)
        "tokens": frames_info,
        "frames": frames_info,
        "notes": {
            "mean": round(float(notes_per.mean()), 2), "min": int(notes_per.min()), "max": int(notes_per.max()),
            "total": int(notes_per.sum()), "hist": [int(c) for c in notes_hist], "hist_cap": max_bins - 1,
            "per_step": round(float(notes_per.sum() / max(1, n * int(HP.width))), 3),
        },
        "steps": {"mean": round(float(steps_per.mean()), 2), "max": int(steps_per.max())},
        "step_hist": [int(c) for c in step_hist],
        "pitch_hist": [int(c) for c in pitch_hist],
        "pitch_labels": pitch_labels(),
        "velocity_hist": [int(c) for c in vel_hist],
        "timing_hist": [int(c) for c in timing_hist],
        "timing_center": int(HP.center_timing),
        "source_files": len(names),
        "bpm": bpm_info,
    }


def stats_summary(stats: dict) -> dict:
    """Small subset of the statistics that is stored inside the dataset info."""
    if not stats or "notes" not in stats:
        return {}
    bpm = stats.get("bpm") or {}
    return {
        "n_sequences": stats["n_sequences"], "unique_sequences": stats["unique_sequences"],
        "source_files": stats["source_files"], "notes_mean": stats["notes"]["mean"],
        "tokens_mean": stats["tokens"]["mean"], "bpm_min": bpm.get("min"), "bpm_max": bpm.get("max"),
        "bpm_mean": bpm.get("mean"), "max_notes": stats.get("max_notes"),
        "truncated_sequences": stats.get("truncated_sequences"),
    }


def dataset_info(name: str) -> dict:
    ds = load_dataset_file(name)
    with _load_lock:
        stats = _stats_cache.get(ds["key"])
    if stats is None:
        stats = compute_stats(ds["data"], ds["metadata"])
        with _load_lock:
            _stats_cache.clear()
            _stats_cache[ds["key"]] = stats
    path: Path = ds["path"]
    stat = path.stat()
    token_format = dataset_format(ds["data"].shape, ds["info"])
    return {
        "name": ds["name"],
        "path": str(path),
        "size": int(stat.st_size),
        "shape": [int(s) for s in ds["data"].shape],
        "dtype": str(ds["data"].dtype),
        "has_metadata": bool(ds["metadata"]),
        "info": _complete_info(ds["info"], ds["name"], path),
        "stats": stats,
        "token_format": token_format,
        "compatible": token_format == compound.TOKEN_FORMAT,
        "legacy_message": "" if token_format == compound.TOKEN_FORMAT else f"{ds['name']}: {LEGACY_MESSAGE}.",
        "hparams": {
            "width": int(HP.width), "n_timing": int(HP.n_timing), "n_vel": int(HP.n_vel),
            "n_pitch": int(HP.n_pitch), "max_seq_len": int(HP.max_seq_len),
        },
    }


# ---------------------------------------------------------------------------------------------- sequence view
def parse_notes(seq) -> list[dict]:
    """Frame sequence [T, 4] -> note list for piano roll and playback (legacy 2D sequences give no notes)."""
    frames = np.asarray(seq)
    if frames.ndim != 2 or frames.shape[-1] != compound.N_FIELDS:
        return []
    notes = compound.frames_to_notes(frames, HP)
    for note in notes:
        note["generic"] = int(chromatic_to_generic(note["pitch"]))
    return notes


def sequence_view(name: str, index: int) -> dict:
    ds = load_dataset_file(name)
    data = ds["data"]
    count = int(data.shape[0])
    if count == 0:
        raise ValueError("The dataset is empty.")
    idx = max(0, min(int(index), count - 1))
    seq = data[idx]
    legacy = not compound.is_compound_array(data)
    if legacy:
        frames: list = []
    else:
        frames = np.asarray(seq)[:compound.sequence_length(seq)].astype(int).tolist()
    meta = ds["metadata"][idx] if idx < len(ds["metadata"]) else None
    return {
        "name": ds["name"],
        "index": idx,
        "count": count,
        "filename": _meta_name(meta, ""),
        "window": _meta_window(meta),
        "bpm": _meta_bpm(meta),
        "steps": int(HP.width),
        "n_pitch": int(HP.n_pitch),
        "notes": parse_notes(seq),
        "legacy": legacy,
        "token_format": compound.LEGACY_FORMAT if legacy else compound.TOKEN_FORMAT,
        # frames [step, pitch, velocity, deviation] (indexed) up to and including EOS
        "n_tokens": len(frames),
        "tokens": frames[:512],
    }


# ---------------------------------------------------------------------------------------------- jobs
class _Cancelled(Exception):
    pass


def _num(raw: dict, key: str, errors: dict, default, lo=None, hi=None, integer=False):
    value = raw.get(key, default)
    if value is None or value == "":
        value = default
    try:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError
        if integer:
            if value != int(value):
                raise ValueError
            value = int(value)
    except (TypeError, ValueError):
        errors[key] = "Enter an integer." if integer else "Enter a number."
        return default
    if lo is not None and value < lo:
        errors[key] = f"Must be >= {lo}."
    elif hi is not None and value > hi:
        errors[key] = f"Must be <= {hi}."
    return value


def _flag(raw: dict, key: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _check_output_name(raw: dict, errors: dict, forbidden: list[str] | None = None):
    name, error = validate_dataset_name(raw.get("name"))
    overwrite = _flag(raw, "overwrite")
    if error:
        errors["name"] = error
    elif forbidden and name in forbidden:
        errors["name"] = "The output must differ from the source dataset(s)."
    elif (data_dir() / name).exists() and not overwrite:
        errors["name"] = f"'{name}' already exists. Enable 'overwrite' or choose another name."
    return name, overwrite


def validate_extract(raw) -> tuple[dict, dict]:
    raw = raw if isinstance(raw, dict) else {}
    errors: dict = {}
    folder = str(raw.get("folder") or "").strip()
    if not folder:
        errors["folder"] = "Choose a MIDI folder on the server."
    elif not Path(folder).expanduser().is_dir():
        errors["folder"] = "Folder does not exist on the server."
    name, overwrite = _check_output_name(raw, errors)
    mapping = str(raw.get("mapping") or "none")
    if mapping not in MAPPING_CHOICES:
        errors["mapping"] = "Unknown mapping."
    params = {
        "folder": str(Path(folder).expanduser()) if folder else "",
        "name": name, "overwrite": overwrite, "mapping": mapping,
        "skip": _num(raw, "skip", errors, 0, 0, integer=True),
        "take": _num(raw, "take", errors, 0, 0, integer=True),
        "jump": _num(raw, "jump", errors, 1, 1, 64, integer=True),
        "randomize_velocity": _num(raw, "randomize_velocity", errors, 0.0, 0.0, 1.0),
        "qnt_strength": _num(raw, "qnt_strength", errors, 0.0, 0.0, 1.0),
        "qnt_randomness": _num(raw, "qnt_randomness", errors, 0.0, 0.0, 1.0),
        "clip": _flag(raw, "clip"), "drums_only": _flag(raw, "drums_only"),
        "filter_duplicates": _flag(raw, "filter_duplicates", True), "pitch_randomness": _flag(raw, "pitch_randomness"),
        "roland": _flag(raw, "roland", True), "remap": _flag(raw, "remap"),
    }
    if mapping == "generic_midi_mapping" and params["drums_only"]:
        pass  # allowed, same as data.py
    return params, errors


def validate_merge(raw) -> tuple[dict, dict]:
    raw = raw if isinstance(raw, dict) else {}
    errors: dict = {}
    sources = raw.get("sources")
    clean_sources: list[str] = []
    if not isinstance(sources, (list, tuple)) or len(sources) < 2:
        errors["sources"] = "Select at least two datasets to merge."
    else:
        for source in sources:
            path, error = existing_dataset_path(source)
            if error:
                errors["sources"] = error
                break
            if path.name not in clean_sources:
                clean_sources.append(path.name)
        if "sources" not in errors and len(clean_sources) < 2:
            errors["sources"] = "Select at least two different datasets."
        if "sources" not in errors:
            formats = {source: dataset_format_of(source) for source in clean_sources}
            legacy = [s for s, (fmt, _) in formats.items() if fmt != compound.TOKEN_FORMAT]
            lengths = {shape[1] for fmt, shape in formats.values() if fmt == compound.TOKEN_FORMAT and shape}
            if legacy:
                errors["sources"] = "Cannot merge legacy (triplet) datasets: " + ", ".join(legacy) + f" ({LEGACY_MESSAGE})."
            elif len(lengths) > 1:
                errors["sources"] = "The datasets have different max_seq_len (" + ", ".join(str(x) for x in sorted(lengths)) + " frames)."
    name, overwrite = _check_output_name(raw, errors, clean_sources)
    return {"sources": clean_sources, "name": name, "overwrite": overwrite, "dedupe": _flag(raw, "dedupe")}, errors


def validate_augment(raw) -> tuple[dict, dict]:
    raw = raw if isinstance(raw, dict) else {}
    errors: dict = {}
    source, error = existing_dataset_path(raw.get("source"))
    if error:
        errors["source"] = error
    elif dataset_format_of(source.name)[0] != compound.TOKEN_FORMAT:
        errors["source"] = f"{source.name}: {LEGACY_MESSAGE}."
    name, overwrite = _check_output_name(raw, errors, [source.name] if source else None)

    def _list(key, caster, lo, hi, default):
        values = raw.get(key, default)
        if isinstance(values, str):
            values = [v for v in re.split(r"[,;\s]+", values.strip()) if v]
        out = []
        try:
            for value in values:
                value = caster(float(value)) if caster is int and float(value) == int(float(value)) else caster(value)
                if value < lo or value > hi:
                    raise ValueError
                out.append(value)
        except (TypeError, ValueError):
            errors[key] = f"Enter comma separated numbers between {lo} and {hi}."
            return list(default)
        if not out:
            errors[key] = "Enter at least one value."
        return out[:12]

    params = {
        "source": source.name if source else "", "name": name, "overwrite": overwrite,
        "velocity_scales": _list("velocity_scales", float, 0.05, 4.0, (0.5, 0.75, 1.25)),
        "microshift_deltas": _list("microshift_deltas", int, -16, 16, (-2, 2)),
        "keep_percentage": _num(raw, "keep_percentage", errors, 100.0, 0.1, 100.0),
    }
    seed = raw.get("seed")
    if seed in (None, ""):
        params["seed"] = None
    else:
        params["seed"] = _num(raw, "seed", errors, 0, 0, 2**31 - 1, integer=True)
    return params, errors


class DatasetJobManager:
    """Runs one dataset job (extract / merge / augment) at a time in a background thread."""

    KINDS = ("extract", "merge", "augment")
    VALIDATORS = {"extract": validate_extract, "merge": validate_merge, "augment": validate_augment}

    def __init__(self, on_finish=None, max_logs: int = 300):
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._on_finish = on_finish
        self._max_logs = max_logs
        self.version = 0
        self._reset()

    # -------------------------------------------------------------- state
    def _reset(self):
        self._state = "idle"
        self._kind = None
        self._params: dict = {}
        self._phase = ""
        self._progress = 0.0
        self._done = 0
        self._total = 0
        self._message = ""
        self._stats: dict = {}
        self._logs: deque = deque(maxlen=self._max_logs)
        self._result: dict | None = None
        self._error: str | None = None
        self._started = 0.0
        self._finished = 0.0

    def _bump(self):
        self.version += 1

    def _set(self, **fields):
        with self._lock:
            for key, value in fields.items():
                setattr(self, f"_{key}", value)
            self._bump()

    def _log(self, text: str):
        with self._lock:
            self._logs.append({"t": time.strftime("%H:%M:%S"), "text": str(text)})
            self._bump()

    def is_active(self) -> bool:
        with self._lock:
            return self._state in ("running", "cancelling")

    def snapshot(self) -> dict:
        with self._lock:
            end = self._finished if self._finished else time.time()
            elapsed = max(0.0, end - self._started) if self._started else 0.0
            eta = None
            if self._state == "running" and self._progress > 0.03:
                eta = elapsed / self._progress * (1.0 - self._progress)
            return {
                "version": self.version,
                "started": self._started,
                "state": self._state,
                "kind": self._kind,
                "params": dict(self._params),
                "phase": self._phase,
                "progress": round(float(self._progress), 4),
                "done": self._done,
                "total": self._total,
                "message": self._message,
                "stats": dict(self._stats),
                "logs": list(self._logs),
                "result": dict(self._result) if self._result else None,
                "error": self._error,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1) if eta is not None else None,
            }

    # -------------------------------------------------------------- control
    def start(self, kind: str, raw) -> tuple[bool, dict]:
        if kind not in self.KINDS:
            return False, {"general": f"Unknown job '{kind}'."}
        with self._lock:
            if self._state in ("running", "cancelling"):
                return False, {"general": "Another dataset job is already running."}
        params, errors = self.VALIDATORS[kind](raw)
        if errors:
            return False, errors
        with self._lock:
            self._reset()
            self._state = "running"
            self._kind = kind
            self._params = dict(params)
            self._phase = "starting"
            self._started = time.time()
            self._cancel.clear()
            self._thread = threading.Thread(target=self._run, args=(kind, params), name=f"dataset-{kind}", daemon=True)
            self._bump()
            self._thread.start()
        return True, {}

    def cancel(self) -> bool:
        with self._lock:
            if self._state != "running":
                return False
            self._state = "cancelling"
            self._message = "Cancelling …"
            self._bump()
        self._cancel.set()
        return True

    def wait(self, timeout: float = 60.0) -> bool:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _check_cancel(self):
        if self._cancel.is_set():
            raise _Cancelled()

    # -------------------------------------------------------------- runner
    def _run(self, kind: str, params: dict):
        final_state = "finished"
        try:
            runner = getattr(self, f"_run_{kind}")
            result = runner(params)
            with self._lock:
                self._result = result
                self._progress = 1.0
                self._phase = "done"
                self._message = f"Saved {result['name']} ({result['n_sequences']} sequences)"
            self._log(f"Saved {result['path']}")
        except _Cancelled:
            final_state = "cancelled"
            self._log("Cancelled. Nothing was written.")
        except BaseException as exc:  # includes SystemExit raised by exit() calls inside the extractor
            if type(exc).__name__ in ("ExtractionCancelled", "AugmentationCancelled"):
                final_state = "cancelled"
                self._log("Cancelled. Nothing was written.")
            else:
                final_state = "failed"
                traceback.print_exc()
                message = str(exc) or type(exc).__name__
                with self._lock:
                    self._error = message
                self._log(f"ERROR: {message}")
        with self._lock:
            self._state = final_state
            self._finished = time.time()
            if final_state == "cancelled":
                self._message = "Cancelled"
            self._bump()
        if self._on_finish is not None:
            try:
                self._on_finish(final_state)
            except Exception:
                traceback.print_exc()

    def _run_extract(self, params: dict) -> dict:
        from modules.midi_processor import MidiProcessor

        started = time.time()
        folder = params["folder"]
        self._set(phase="scanning", message=f"Scanning {folder}")
        scan = scan_midi_folder(folder, max_seconds=60.0)
        if not scan.get("ok") or scan["midi_files"] == 0:
            raise ValueError(f"No MIDI files found in {folder}")
        self._log(f"Found {scan['midi_files']} MIDI files in {folder}")
        self._check_cancel()

        def on_progress(idx, n_files, filename, stats, n_sequences):
            with self._lock:
                self._phase = "extracting" if idx < n_files else "finalizing"
                self._done, self._total = idx, n_files
                self._progress = 0.9 * (idx / n_files) if n_files else 0.0
                self._message = (os.path.basename(filename) if filename else "Filtering duplicates") or ""
                self._stats = {
                    "sequences": int(n_sequences), "skipped_invalid": stats.skipped_invalid,
                    "skipped_time_signature": stats.skipped_time_signature, "skipped_no_drums": stats.skipped_no_drums,
                    "skipped_too_short": stats.skipped_too_short, "one_bar_patterns": stats.one_bar_patterns,
                    "truncated_sequences": stats.truncated_sequences, "merged_duplicates": stats.merged_duplicates,
                    "max_notes_per_window": stats.max_notes_per_window,
                }
                self._bump()

        processor = MidiProcessor(HP)
        mapping = params["mapping"]
        processor.extract(
            midi_folder_path=folder, skip=params["skip"], take=params["take"], clip=params["clip"],
            randomize_velocity=params["randomize_velocity"], qnt_strength=params["qnt_strength"],
            qnt_randomness=params["qnt_randomness"], pitch_randomness=params["pitch_randomness"],
            roland=params["roland"], drums_only=params["drums_only"], filter_duplicates=params["filter_duplicates"],
            remap=params["remap"], jump=params["jump"],
            generic_midi_mapping=mapping == "generic_midi_mapping",
            groove_to_specific=mapping == "groove_to_specific",
            generic_to_chromatic=mapping == "generic_to_chromatic",
            progress_callback=on_progress, should_stop=self._cancel.is_set,
        )
        self._check_cancel()
        stats = processor.last_stats
        for failed, error in stats.failed_files[:20]:
            self._log(f"Failed: {os.path.basename(failed)}: {error}")
        if not processor.sequences:
            raise ValueError("No sequences could be extracted (see the skipped-file counters).")

        self._set(phase="saving", progress=0.92, message="Writing dataset")
        padded = pad_frames(processor.sequences)
        metadata = [_normalize_meta_entry(m, i) for i, m in enumerate(processor.metadata)]
        if stats.truncated_sequences:
            self._log(
                f"{stats.truncated_sequences} window(s) had more than {int(HP.max_seq_len) - 1} notes and were cut "
                f"(max {stats.max_notes_per_window} notes per window)"
            )
        if stats.merged_duplicates:
            self._log(f"Merged {stats.merged_duplicates} duplicate note(s) with the same step and pitch")
        summary = stats_summary(compute_stats(padded, metadata))
        info = _base_info(params["name"], "extract")
        info["source"] = {"folder": folder, "midi_files": scan["midi_files"]}
        info["options"] = {k: params[k] for k in (
            "mapping", "skip", "take", "jump", "clip", "drums_only", "filter_duplicates", "randomize_velocity",
            "qnt_strength", "qnt_randomness", "pitch_randomness", "roland", "remap")}
        info["stats"] = dict(summary, extraction={
            "files_in_selection": stats.files_count, "skipped_invalid": stats.skipped_invalid,
            "skipped_time_signature": stats.skipped_time_signature, "skipped_no_drums": stats.skipped_no_drums,
            "skipped_too_short": stats.skipped_too_short, "one_bar_patterns": stats.one_bar_patterns,
            "sequences_before_filter": stats.sequences_before_filter, "failed_files": len(stats.failed_files),
            "truncated_sequences": stats.truncated_sequences, "merged_duplicates": stats.merged_duplicates,
            "max_notes_per_window": stats.max_notes_per_window,
            "seconds": round(time.time() - started, 1),
        })
        info["description"] = f"MIDI extraction from {os.path.basename(folder.rstrip('/')) or folder}"
        path = save_dataset(params["name"], padded, metadata, info, overwrite=params["overwrite"])
        return {"name": path.name, "path": str(path), "n_sequences": int(padded.shape[0])}

    def _run_merge(self, params: dict) -> dict:
        sources = params["sources"]
        arrays, metas, source_infos = [], [], []
        for i, source in enumerate(sources):
            self._check_cancel()
            self._set(phase="loading", done=i, total=len(sources), progress=0.7 * i / len(sources), message=source)
            ds = load_dataset_file(source)
            if not is_compatible(ds["data"].shape, ds["info"]):
                raise ValueError(f"{source}: {LEGACY_MESSAGE}.")
            arrays.append(np.asarray(ds["data"], dtype=np.int16))
            metas.append([_normalize_meta_entry(m, j) for j, m in enumerate(ds["metadata"])] if ds["metadata"]
                         else [(f"{source}:{j}", 0) for j in range(len(ds["data"]))])
            source_infos.append({"name": source, "n_sequences": int(len(ds["data"])), "kind": ds["info"].get("kind", "legacy"),
                                 "description": ds["info"].get("description", "")})
            self._log(f"Loaded {source}: {arrays[-1].shape}")
        self._check_cancel()

        lengths = {int(a.shape[1]) for a in arrays}
        if len(lengths) > 1:
            raise ValueError("The datasets have different max_seq_len: " + ", ".join(str(x) for x in sorted(lengths)))
        merged = np.concatenate(arrays, axis=0).astype(np.int16)
        merged_meta = [entry for entries in metas for entry in entries]

        removed = 0
        if params["dedupe"]:
            self._set(phase="deduplicating", progress=0.8, message="Removing duplicate sequences")
            seen, keep = set(), []
            rows = np.ascontiguousarray(merged).reshape(len(merged), -1)
            for i in range(len(rows)):
                key = rows[i].tobytes()
                if key not in seen:
                    seen.add(key)
                    keep.append(i)
            removed = len(rows) - len(keep)
            merged = merged[keep]
            merged_meta = [merged_meta[i] for i in keep]
            self._log(f"Removed {removed} duplicate sequences")
        self._check_cancel()

        self._set(phase="saving", progress=0.9, message="Writing dataset")
        info = _base_info(params["name"], "merge")
        info["source"] = {"datasets": source_infos}
        info["options"] = {"dedupe": params["dedupe"]}
        info["stats"] = dict(stats_summary(compute_stats(merged, merged_meta)), duplicates_removed=removed)
        info["description"] = "Merge of " + ", ".join(sources)
        path = save_dataset(params["name"], merged, merged_meta, info, overwrite=params["overwrite"])
        return {"name": path.name, "path": str(path), "n_sequences": int(merged.shape[0])}

    def _run_augment(self, params: dict) -> dict:
        import data_augment

        ds = load_dataset_file(params["source"])
        self._set(phase="augmenting", message=params["source"], total=int(len(ds["data"])))

        def on_progress(done, total):
            with self._lock:
                self._done, self._total = done, total
                self._progress = 0.9 * (done / total) if total else 0.0
                self._bump()

        padded, meta, stats = data_augment.build_augmented_dataset(
            ds["data"], ds["metadata"] if ds["metadata"] else None,
            velocity_scales=tuple(params["velocity_scales"]), microshift_deltas=tuple(params["microshift_deltas"]),
            keep_percentage=params["keep_percentage"], progress_callback=on_progress,
            should_stop=self._cancel.is_set, seed=params["seed"],
        )
        self._check_cancel()
        self._log(f"Generated {stats['generated_unique']} unique variants, keeping {stats['output_sequences']}")
        self._set(phase="saving", progress=0.93, message="Writing dataset")
        info = _base_info(params["name"], "augment")
        info["source"] = {"dataset": params["source"], "n_sequences": stats["source_sequences"]}
        info["options"] = {k: params[k] for k in ("velocity_scales", "microshift_deltas", "keep_percentage", "seed")}
        info["stats"] = dict(stats_summary(compute_stats(padded, meta)), generated_unique=stats["generated_unique"])
        info["description"] = f"Augmentation of {params['source']}"
        path = save_dataset(params["name"], padded, meta, info, overwrite=params["overwrite"])
        return {"name": path.name, "path": str(path), "n_sequences": int(padded.shape[0])}
