import torch
from tiny_gptv.blocks import TinyGPTVBlock

x = torch.rand(2, 8, 512)
lora_mha = TinyGPTVBlock(512, 8)
out = lora_mha(x)
print(out.shape)
