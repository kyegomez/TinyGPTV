[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# TinyGPTV
Simple Implementation of TinyGPTV in super simple Zeta lego blocks. Here all the modules from figure 2 are implemented in Zeta and Pytorch

## Usage
```python
import torch
from tiny_gptv import TinyGPTVBlock

x = torch.rand(2, 8, 512)
lora_mha = TinyGPTVBlock(512, 8)
out = lora_mha(x)
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