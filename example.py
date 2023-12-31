import torch
from tiny_gptv.blocks import TinyGPTVBlock

# Random tensor, replace with your input data
x = torch.rand(2, 8, 512)

# TinyGPTVBlock
block = TinyGPTVBlock(512, 8, depth=10)

# Print the block
print(block)

# Forward pass
out = block(x)

# Print the output shape
print(out.shape)
