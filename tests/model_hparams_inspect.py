from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import argparse

import torch


# DEFAULT_MODEL_PATH = Path("models/dev/model_2026-04-08_09-52-22.pt")
DEFAULT_MODEL_PATH = Path("models/exp/model_2026-04-21_22-31-51.pt")

def _print_hparams_summary(hparams: SimpleNamespace, model_path: Path) -> None:
	keys = [
		"dataset_name",
		"validation_split",
		"n_samples",
		"seed",
		"batch_size",
		"n_epochs",
		"learning_rate",
		"scheduler_start_lr",
		"scheduler_end_lr",
		"weight_decay",
		"latent_dim",
		"d_model",
		"n_layers",
		"n_heads",
		"max_seq_len",
	]

	print("Checkpoint:", model_path)
	print("Saved hparams (selected):")
	for key in keys:
		print(f"- {key}: {getattr(hparams, key, None)}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect saved hparams from a .pt model checkpoint.")
	parser.add_argument(
		"--model",
		type=Path,
		default=DEFAULT_MODEL_PATH,
		help="Path to model checkpoint (.pt).",
	)
	args = parser.parse_args()

	model_path = args.model
	if not model_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {model_path}")

	checkpoint = torch.load(str(model_path), map_location="cpu")
	if "hparams" not in checkpoint:
		raise KeyError(f"No 'hparams' key in checkpoint: {model_path}")

	hparams = SimpleNamespace(**checkpoint["hparams"])
	_print_hparams_summary(hparams, model_path)


if __name__ == "__main__":
	main()
