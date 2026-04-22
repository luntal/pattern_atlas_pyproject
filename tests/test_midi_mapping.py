from pathlib import Path
import csv


def _is_used(value: str) -> bool:
	if value is None:
		return False
	normalized = value.strip().lower()
	if normalized == "":
		return False
	return normalized not in {"0", "false", "no", "n", "none", "nan"}


def print_midi_mapping_counts() -> None:
	project_root = Path(__file__).resolve().parents[1]
	csv_path = project_root / "static" / "generic_midi_mapping.csv"

	used_count = 0
	sound_technique_present_count = 0

	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if _is_used(row.get("Global Grooves Instruments", "")):
				used_count += 1

			if (row.get("Sound/Technique", "") or "").strip():
				sound_technique_present_count += 1

	print(f"Global Grooves instruments used: {used_count}")
	print(f"Instruments present in Sound/Technique column: {sound_technique_present_count}")


if __name__ == "__main__":
	print_midi_mapping_counts()
