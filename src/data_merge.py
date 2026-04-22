import numpy as np
from pathlib import Path
from hparams import hparams
from modules.branch_paths import branch_dir, get_current_branch,safe_branch_name

current_branch = safe_branch_name(get_current_branch())
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / current_branch


# List of .npz files to merge
files_to_merge = [
    DATA_DIR / 'dataset-groove.npz',
    DATA_DIR / 'dataset-groove-augmented.npz',
    DATA_DIR / 'dataset-groove-augmented-rnd.npz',
    DATA_DIR / 'dataset-lakh.npz',
    # DATA_DIR / 'dataset-lakh-augmented.npz',
]

save_path = DATA_DIR / 'dataset-merged.npz'

# Load all datasets
all_data = []
all_metadata = []

for file_path in files_to_merge:
    npz_file = np.load(file_path, allow_pickle=True)
    data = npz_file['data']
    all_data.append(data)
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
if merged_metadata is not None:
    np.savez(save_path, data=merged_data, metadata=merged_metadata)
else:
    np.savez(save_path, data=merged_data)
print(f"\nData saved to {save_path}")
