"""Compound token dataset augmentation utility.

Loads compound frame sequences ([N, T, 4] = step, pitch, velocity, deviation) from an NPZ file in the
data directory, applies several augmentation strategies directly on the note fields, and saves the
augmented dataset back to NPZ (frames are encoded with modules.compound).

Augmentation types:
- Random note removal: remove up to three notes at a time and save each step
	until only a random minimum of 4-8 notes remains.
- Pitch-line removal: for each pitch, remove notes in pairs (two at a time),
	saving after each removal step.
- Global velocity + microshift variants: create additional versions of each
	generated sequence using one shared velocity scale and one shared timing
	shift across the full sequence.

Output balancing:
- The saved dataset contains only augmented sequences.
- Augmented output is capped to the original dataset size.
"""

from pathlib import Path
from typing import List, Tuple
import random
import time

import numpy as np

from hparams import hparams
from modules import compound
from modules.branch_paths import latest_in_branch, safe_branch_name, get_current_branch

# A note is a raw tuple (step, pitch, velocity, deviation); a sequence is a sorted list of notes.
Note = Tuple[int, int, int, int]



def _project_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
	branch_name = safe_branch_name(get_current_branch())
	return _project_root() / "data" / branch_name


def _load_npz_from_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
	file_path = _data_dir() / filename
	if not file_path.exists():
		raise FileNotFoundError(f"Dataset not found: {file_path}")

	npz_data = np.load(file_path, allow_pickle=True)
	if "data" in npz_data:
		sequences = npz_data["data"]
	elif "tensor" in npz_data:
		sequences = npz_data["tensor"]
	else:
		raise ValueError("NPZ must contain key 'data' or 'tensor'.")

	metadata = npz_data["metadata"] if "metadata" in npz_data else None
	return sequences, metadata


def _trim_sequence(seq: np.ndarray) -> List[Note]:
	"""Frame sequence [T, 4] -> sorted list of raw notes (up to EOS)."""
	return [tuple(int(x) for x in row) for row in compound.frames_to_raw(seq, hparams).tolist()]


def _random_note_removals(base_seq: List[Note]) -> List[List[Note]]:
	results: List[List[Note]] = []
	current = base_seq[:]
	n_notes = len(current)
	if n_notes <= 4:
		return results

	target_min = random.randint(4, 8)
	target = min(target_min, n_notes)

	while len(current) > target:
		max_removable = len(current) - target
		remove_count = min(3, max_removable)
		if remove_count <= 0:
			break
		remove_idxs = set(random.sample(range(len(current)), remove_count))
		current = [note for i, note in enumerate(current) if i not in remove_idxs]
		results.append(current[:])

	return results


def _pitch_line_removals(base_seq: List[Note]) -> List[List[Note]]:
	results: List[List[Note]] = []
	unique_pitches = sorted({note[1] for note in base_seq})

	for pitch in unique_pitches:
		current = base_seq[:]
		while True:
			pitch_ordinals = [i for i, note in enumerate(current) if note[1] == pitch]
			if len(pitch_ordinals) < 3:
				break

			remove_group = set(pitch_ordinals[:3])
			current = [note for i, note in enumerate(current) if i not in remove_group]
			results.append(current[:])

	return results


def _apply_global_velocity_and_microshift(
	seq_notes: List[Note],
	velocity_scale: float,
	microshift_delta: int,
	velocity_jitter: int = 1,
	microshift_jitter: int = 1,
) -> List[Note]:
	out: List[Note] = []
	for step, pitch, v, t in seq_notes:
		v_new = int(round(v * velocity_scale))
		if velocity_jitter > 0:
			v_new += random.randint(-velocity_jitter, velocity_jitter)
		v_new = max(0, min(v_new, hparams.n_vel - 1))
		t_new = t + microshift_delta
		if microshift_jitter > 0:
			t_new += random.randint(-microshift_jitter, microshift_jitter)
		t_new = max(0, min(t_new, hparams.n_timing - 1))
		out.append((step, pitch, v_new, t_new))
	return out


def _to_padded_array(sequences: List[List[Note]], max_len: int) -> np.ndarray:
	"""Note lists -> int16 frames [N, max_len, 4] (EOS after the notes, PAD after EOS)."""
	arr = np.zeros((len(sequences), max_len, compound.N_FIELDS), dtype=np.int16)
	for i, seq in enumerate(sequences):
		arr[i] = compound.notes_to_frames(seq, hparams, max_len=max_len)
	return arr


def _normalize_metadata_entry(raw_meta, fallback_idx: int) -> Tuple:
	"""Return metadata as a flat (filename, bar_index[, bpm, ...]) tuple.

	Additional entries after the bar index (e.g. the BPM) are preserved.
	"""
	filename = str(fallback_idx)
	bar_index = 0
	extra: Tuple = ()

	if raw_meta is None:
		return filename, bar_index

	if isinstance(raw_meta, np.ndarray):
		raw_meta = raw_meta.tolist()

	if isinstance(raw_meta, (list, tuple)):
		if len(raw_meta) > 0:
			filename = str(raw_meta[0])
		if len(raw_meta) > 1:
			try:
				bar_index = int(raw_meta[1])
			except (TypeError, ValueError):
				bar_index = 0
		if len(raw_meta) > 2:
			extra = tuple(raw_meta[2:])
	else:
		filename = str(raw_meta)

	return (filename, bar_index) + extra


class AugmentationCancelled(Exception):
	"""Raised by build_augmented_dataset when should_stop() returns True."""


def build_augmented_dataset(
	data: np.ndarray,
	metadata,
	velocity_scales: Tuple[float, ...] = (0.5, 0.75, 1.25),
	microshift_deltas: Tuple[int, ...] = (-2, 2),
	keep_percentage: float = 100.0,
	progress_callback=None,
	should_stop=None,
	seed=None,
):
	"""Pure computation part of the augmentation (no file access). ``data`` holds compound frames [N, T, 4].

	Returns (padded_array, metadata_list, stats_dict).
	progress_callback(done, total) is called once per source sequence.
	should_stop() -> bool is polled per source sequence and raises AugmentationCancelled.
	"""
	if keep_percentage < 0 or keep_percentage > 100:
		raise ValueError("keep_percentage must be between 0 and 100.")
	if seed is not None:
		random.seed(seed)
	if not compound.is_compound_array(data):
		raise ValueError(
			"Augmentation needs a compound dataset [N, T, 4]; legacy triplet datasets must be re-extracted in the Dataset tab."
		)
	max_len = int(np.shape(data)[1])

	original_sequences: List[List[Note]] = []
	original_meta: List[object] = []
	augmented_sequences: List[List[Note]] = []
	augmented_meta: List[object] = []
	n_total = len(data)
	last_progress_time = time.time()

	for idx, seq in enumerate(data):
		if should_stop is not None and should_stop():
			raise AugmentationCancelled()
		if progress_callback is not None:
			progress_callback(idx, n_total)
		now = time.time()
		if now - last_progress_time >= 5:
			print(f"Processed {idx}/{n_total} source files")
			last_progress_time = now

		base_seq = _trim_sequence(seq)
		raw_meta = metadata[idx] if metadata is not None and idx < len(metadata) else None
		meta_entry = _normalize_metadata_entry(raw_meta, idx)
		filename, bar_index, meta_tail = meta_entry[0], meta_entry[1], tuple(meta_entry[2:])

		original_sequences.append(base_seq)
		original_meta.append((filename, bar_index) + meta_tail)

		generated = _random_note_removals(base_seq)
		generated.extend(_pitch_line_removals(base_seq))

		for gen_seq in generated:
			augmented_sequences.append(gen_seq)
			augmented_meta.append((filename, bar_index) + meta_tail)

			variation_specs = [(v_scale, t_delta) for v_scale in velocity_scales for t_delta in microshift_deltas]
			n_keep = max(1, len(variation_specs) // 2)
			kept_specs = random.sample(variation_specs, n_keep)

			for v_scale, t_delta in kept_specs:
				varied = _apply_global_velocity_and_microshift(gen_seq, v_scale, t_delta)
				augmented_sequences.append(varied)
				augmented_meta.append((f"{filename} | v{v_scale}_t{t_delta}", bar_index) + meta_tail)

	if not augmented_sequences:
		raise ValueError("No augmented sequences were generated.")

	print(f"Processed {n_total}/{n_total} source files")

	unique_aug_sequences = []
	unique_aug_meta = []
	seen = set()
	for i, seq in enumerate(augmented_sequences):
		key = tuple(seq)
		if key in seen:
			continue
		seen.add(key)
		unique_aug_sequences.append(seq)
		unique_aug_meta.append(augmented_meta[i])

	if not original_sequences:
		raise ValueError("No original sequences were loaded.")

	# Keep only augmented data, capped to original dataset size.
	aug_count = min(len(original_sequences), len(unique_aug_sequences))
	if aug_count == 0:
		raise ValueError("Unable to build augmented dataset because no unique augmentations were generated.")

	aug_indices = random.sample(range(len(unique_aug_sequences)), aug_count)
	selected_aug_sequences = [unique_aug_sequences[i] for i in aug_indices]
	selected_aug_meta = [unique_aug_meta[i] for i in aug_indices]

	if keep_percentage < 100:
		n_keep = int(round(aug_count * (keep_percentage / 100.0)))
		n_keep = max(1, min(n_keep, aug_count))
		keep_indices = random.sample(range(aug_count), n_keep)
		selected_aug_sequences = [selected_aug_sequences[i] for i in keep_indices]
		selected_aug_meta = [selected_aug_meta[i] for i in keep_indices]

	final_sequences = selected_aug_sequences
	final_meta = selected_aug_meta

	padded = _to_padded_array(final_sequences, max_len)
	if progress_callback is not None:
		progress_callback(n_total, n_total)
	stats = {
		"source_sequences": len(original_sequences),
		"generated_unique": len(unique_aug_sequences),
		"output_sequences": len(final_sequences),
		"keep_percentage": keep_percentage,
		"velocity_scales": list(velocity_scales),
		"microshift_deltas": list(microshift_deltas),
	}
	return padded, final_meta, stats


def augment_dataset(
	input_filename: str,
	output_filename: str,
	velocity_scales: Tuple[float, ...] = (0.5, 0.75, 1.25),
	microshift_deltas: Tuple[int, ...] = (-2, 2),
	keep_percentage: float = 100.0,
) -> None:
	data, metadata = _load_npz_from_data(input_filename)
	padded, final_meta, stats = build_augmented_dataset(
		data, metadata, velocity_scales, microshift_deltas, keep_percentage,
	)

	if not output_filename.endswith(".npz"):
		output_filename += ".npz"
	output_path = _data_dir() / output_filename

	import json
	info = dict(compound.format_info(hparams), kind="augment", name=output_filename, max_seq_len=int(padded.shape[1]))
	np.savez(
		output_path,
		data=padded,
		metadata=np.array(final_meta, dtype=object),
		info=np.array(json.dumps(info)),
	)
	print(f"Saved augmented dataset to {output_path}")
	print(f"Sequences: {stats['output_sequences']}")
	print("Saved only augmented sequences (originals are excluded).")
	print(f"Original dataset size cap: {stats['source_sequences']}")
	if keep_percentage < 100:
		print(f"Keep percentage: {keep_percentage}%")
	print(f"Shape: {padded.shape}")


if __name__ == "__main__":
	# Edit these as needed.
	INPUT_FILE = "dataset-groove-rnd.npz"
	OUTPUT_FILE = "dataset-groove-augmented-rnd.npz"
	augment_dataset(input_filename=INPUT_FILE, output_filename=OUTPUT_FILE, keep_percentage=20)
