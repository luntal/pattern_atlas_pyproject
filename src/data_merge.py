import json

import numpy as np
from pathlib import Path
from hparams import hparams
from modules import compound
from modules.branch_paths import branch_dir, get_current_branch,safe_branch_name

current_branch = safe_branch_name(get_current_branch())
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / current_branch


# List of .npz files to merge
files_to_merge = [
    DATA_DIR / 'dataset-groove-selection-2.npz',
    DATA_DIR / 'dataset-global-grooves-2.npz',
    # DATA_DIR / 'dataset-groove.npz',
    # DATA_DIR / 'dataset-groove-augmented.npz',
    # DATA_DIR / 'dataset-groove-augmented-rnd.npz',
    # DATA_DIR / 'dataset-lakh.npz',
    # DATA_DIR / 'dataset-lakh-augmented.npz',
]

save_path = DATA_DIR / 'dataset-merged-global-grooves-2.npz'

# Load all datasets
all_data = []
all_metadata = []

for file_path in files_to_merge:
    npz_file = np.load(file_path, allow_pickle=True)
    data = npz_file['data']
    if not compound.is_compound_array(data):
        raise SystemExit(f"{file_path.name} is a legacy triplet dataset {data.shape}; re-extract it in the Dataset tab.")
    if all_data and data.shape[1] != all_data[0].shape[1]:
        raise SystemExit(f"{file_path.name} has {data.shape[1]} frames per sequence, expected {all_data[0].shape[1]}.")
    all_data.append(data.astype(np.int16))
    print(f"Loaded {file_path}: shape {data.shape}")

    # Load metadata if it exists
    if 'metadata' in npz_file:
        metadata = npz_file['metadata']
        all_metadata.append(metadata)
        print(f"  Metadata entries: {len(metadata)}")
    else:
        print(f"  No metadata found")

# Merge all data
merged_data = np.concatenate(all_data, axis=0)
print(f"\nMerged data shape: {merged_data.shape}")

# Merge all metadata
if all_metadata:
    merged_metadata = np.concatenate(all_metadata, axis=0)
    print(f"Merged metadata entries: {len(merged_metadata)}")
else:
    merged_metadata = None

# Save merged dataset
info = dict(compound.format_info(hparams), kind='merge', name=save_path.name, max_seq_len=int(merged_data.shape[1]))
if merged_metadata is not None:
    np.savez(save_path, data=merged_data, metadata=merged_metadata, info=np.array(json.dumps(info)))
else:
    np.savez(save_path, data=merged_data, info=np.array(json.dumps(info)))
print(f"\nData saved to {save_path}")
