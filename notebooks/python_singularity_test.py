import torch

print(torch.cuda.is_available())

import icosphere

vertices, _ = icosphere.icosphere(4)

print(vertices.shape)
