# import numpy as np
# data = np.load('dataset-groove-merged.npz')['tensor']
# for i in range(15):
#     i+=15000
#     print(data[i])

import sys
import numpy as np
import torch
import torchtune 

print('----------')
print('python version:    ', sys.version)
print('venv path:         ', sys.executable)
print('torch version:     ', torch.__version__)
print('torchtune version: ', torchtune.__version__)
print('----------')
# print('test')
# x = np.array([1, 2, 3])
# print(x)
# y = torch.tensor([1, 2, 3])
# print(y)

import torch
from torch import nn
pos_emb = nn.Embedding(10, 10)
pos_emb = pos_emb(torch.tensor([0,1,2,3,4,5]))
seq = torch.arange(10)
print(seq.shape)
print(seq)
# print(pos_emb.shape)