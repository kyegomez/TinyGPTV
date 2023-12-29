import torch
from torch import nn
from zeta.nn import MLP, Lora, MultiQueryAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g


class TinyGPTVBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dropout=0.1,
        causal=False,
        masked=False,
        masked_seqlen=None,
        expansion_factor=4,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.causal = causal
        self.masked = masked
        self.masked_seqlen = masked_seqlen
        self.dropout = dropout
        self.scale = dim**-0.5
        self.mha = MultiQueryAttention(
            dim, heads
        )
        self.mlp = MLP(
            dim_in=dim,
            dim_out=dim,
            expansion_factor=expansion_factor,
        )
        self.norm = nn.LayerNorm(dim)
        self.lora = Lora(
            dim,
            dim,
            alpha=2
        )
        self.rmsnorm = RMSNorm(dim)

    def forward(self, x: torch.Tensor):
        normed_x = self.norm(x)
        lorad = self.lora(normed_x)
        attn, _, _ = self.mha(normed_x)
        attn_with_lora = attn + lorad
        rms_normed = self.rmsnorm(attn_with_lora)
        mlped_x = self.mlp(x)
        return rms_normed + mlped_x
        
    
