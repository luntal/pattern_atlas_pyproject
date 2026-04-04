import numpy as np
from hparams import hparams

# List of .npz files to merge
files_to_merge = [
    '../data/dataset-groove.npz',
    '../data/dataset-groove-rnd.npz',
    '../data/dataset-lakh.npz'
]

# Load all datasets
all_data = []
all_metadata = []

for file_path in files_to_merge:
    npz_file = np.load(file_path, allow_pickle=True)
    data = npz_file['data']
    if data.shape[0] > 20000:
        data = data[:21000]
    all_data.append(data)
    print(f"Loaded {file_path}: shape {data.shape}")

    # Load metadata if it exists
    if 'metadata' in npz_file:
        metadata = npz_file['metadata']
        if metadata.shape[0] > 20000:
            metadata = metadata[:21000]
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
save_path = '../data/dataset-merged.npz'
if merged_metadata is not None:
    np.savez(save_path, data=merged_data, metadata=merged_metadata)
else:
    np.savez(save_path, data=merged_data)
print(f"\nData saved to {save_path}")
