import torch
from tiny_gptv import LoraMHA

x = torch.rand(2, 8, 512)
block = LoraMHA(512, 8)
out = block(x)
print(out.shape)
