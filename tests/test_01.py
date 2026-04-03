import numpy as np
grid_res = 4
linx = np.linspace(0, 100, grid_res+1)
liny = np.linspace(0, 50, grid_res+1)




def get_centers(lin):
    centers = []
    for i, x in enumerate(lin[:-1]):
        # print(i, x)
        dist = lin[i + 1] - lin[i]
        start = lin[i]
        shift = dist / 2
        center = start + shift
        centers.append(center)
    return centers

c_linx = get_centers(linx)
print(linx)
print(c_linx)
print()
c_liny = get_centers(liny)
print(liny)
print(c_liny)
print()
counter = 0
indices = []
for x in c_linx:
    for y in c_liny:
        # print(x, y)
        indices.append(counter)
        counter += 1
print(len(indices))
