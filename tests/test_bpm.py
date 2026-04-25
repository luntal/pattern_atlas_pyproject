import numpy as np

data = np.load('outputs/global-grooves/field_render_2026-04-25_18-45-58.npz')
bpm = data['bpms_1d']
# print(bpm)
controls = data['control_points']
for i, control in enumerate(controls):
    print(f'{i}: {control}')
