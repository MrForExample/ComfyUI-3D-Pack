import numpy as np
import torch

vector3_tensor = torch.tensor([[1, 2, 3], [3, 2, 1], [2, 3, 1]])  # shape (N, 3)

target_axis = (2, 0, 1) # or [2, 0, 1]
vector3_tensor[:, [0, 1, 2]] = vector3_tensor[:, target_axis]

print(vector3_tensor)
