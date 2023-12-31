[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# TinyGPTV
Simple Implementation of TinyGPTV in super simple Zeta lego blocks. Here all the modules from figure 2 are implemented in Zeta and Pytorch.

The flow is the following:
x -> skip connection -> layer norm -> lora -> mha + lora -> residual_rms_norm -> original_skip_connection -> mlp + rms norm


## Install
`pip3 install tiny-gptv`


## Usage

### TinyGPTVBlock, Figure3 (c):
- Layernorm
- MHA
- Lora
- QK Norm
- RMS Norm
- MLP


```python
import torch
from tiny_gptv.blocks import TinyGPTVBlock

x = torch.rand(2, 8, 512)
block = TinyGPTVBlock(512, 8, depth=10)
out = block(x)
print(out.shape)


```

### Figure3 (b) Lora Module for LLMS Block
- MHA,
- Lora,
- Normalization,
- MLP
- Skip connection
- Split then add

```python
import torch
from tiny_gptv import LoraMHA

x = torch.rand(2, 8, 512)
block = LoraMHA(512, 8)
out = block(x)
print(out.shape)

```


# Citation

```bibtex
@misc{yuan2023tinygptv,
    title={TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones}, 
    author={Zhengqing Yuan and Zhaoxu Li and Lichao Sun},
    year={2023},
    eprint={2312.16862},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```

# License
MIT