import torch
from tiny_gptv.blocks import TinyGPTVBlock

x = torch.rand(2, 8, 512)
block = TinyGPTVBlock(512, 8)
out = block(x)
print(out.shape)
