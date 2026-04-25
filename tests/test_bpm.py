import numpy as np

data = np.load('outputs/global-grooves/field_render_2026-04-25_12-24-04.npz')
bpm = data['bpms_1d']
print(bpm)