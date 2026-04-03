import numpy as np
from modules.model import TransformerVAE
from hparams import hparams
# NUM_POINTS = 4
# l = [266, 108, 66, 102, 366, 598, 715, 204, 110]
# # data_tensor = data[np.random.choice(data.shape[0], NUM_POINTS, replace=False)]
# init_idx = np.random.choice(l, NUM_POINTS, replace=False)
# print(init_idx)


# vae = TransformerVAE(hparams)
# print(vae)

data1 = np.load('dataset.npz')['tensor']
print(data1.shape)

data2 = np.load('dataset-ex1.npz')['tensor']

# Pad data2 to match the length of data1's second dimension
pad_width = data1.shape[1] - data2.shape[1]
if pad_width > 0:
    data2 = np.pad(data2, ((0, 0), (0, pad_width)), constant_values=0)
print(data2.shape)

data3 = np.concatenate([data1[:9996], data2], axis=0)
print(data3.shape)