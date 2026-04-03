import numpy as np
from hparams import hparams
data1 = np.load('dataset-groove-merged.npz')['tensor']
print(data1.shape)
# data1 = data1[:hparams.n_samples]

# merging two datasets
data2 = np.load('dataset-groove-qnt.npz')['tensor']
print(data2.shape)
# Pad data2 to match the length of data1's second dimension
# pad_width = data1.shape[1] - data2.shape[1]
# if pad_width > 0:
#     data2 = np.pad(data2, ((0, 0), (0, pad_width)), constant_values=0)

# data = np.concatenate((data1[:-data2.shape[0]], data2), axis=0)
data = np.concatenate((data1, data2), axis=0)
print(data.shape)

save_path = 'dataset-groove-merged.npz'
np.savez(save_path, tensor=data)
print(f"Data saved to {save_path}")
