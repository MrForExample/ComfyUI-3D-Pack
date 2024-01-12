import os
import numpy as np
import torch

grid_res = 128
p = os.path.join(os.path.dirname(__file__), f'../data/tets/{grid_res}_tets.npz')
print(p)

