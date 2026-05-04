import numpy as np

dataset_path = 'data/global-grooves/dataset-empty.npz'
padded_sequences = np.zeros(256)
padded_sequences[0:33] = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2]
# unsqueeze to add batch dimension
padded_sequences = np.expand_dims(padded_sequences, axis=0)
metadata_array = ['empty.mid', 0, 120]

np.savez(dataset_path, data=padded_sequences, metadata=metadata_array)