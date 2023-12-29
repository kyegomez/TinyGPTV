from typing import Optional

import torch
from torch import nn, Tensor
from zeta.nn import MLP, Lora, MultiQueryAttention
from zeta import enforce_types
from typing import List


class SkipThenAdd(nn.Module):
    def __init__(
        self,
        modules: List[nn.Module] = None,
        *args,
        **kwargs
    ):
        """
        Initializes a SkipThenAdd module.

        Args:
            modules (List[nn.Module], optional): List of modules to be applied to the input. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.modules = modules
        
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the SkipThenAdd module.

        Args:
            x (Tensor): Input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: Output tensor.
        """
        for module in self.modules:
            module = module(x, *args, **kwargs)
            out = module + x
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class LoraMHABlock(nn.Module):
    """
    LoraMHA is a module that combines Lora and MultiQueryAttention layers.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        masked (bool, optional): Whether to use masked attention. Defaults to False.
        masked_seqlen (int, optional): The length of the masked sequence. Defaults to None.
        expansion_factor (int, optional): The expansion factor for the MLP. Defaults to 4.
    """

    @enforce_types
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        causal: bool = False,
        masked: bool = False,
        masked_seqlen: Optional[int] = None,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.causal = causal
        self.masked = masked
        self.masked_seqlen = masked_seqlen
        self.dropout = dropout
        self.scale = dim**-0.5
        self.mha = MultiQueryAttention(dim, heads)
        self.mlp = MLP(
            dim_in=dim, dim_out=dim, expansion_factor=expansion_factor
        )
        self.norm = nn.LayerNorm(dim)
        self.lora = Lora(dim, dim, alpha=2)

    @enforce_types
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoraMHA module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        to_mha, _, _ = self.mha(x)
        to_lora = self.lora(x)
        add_together = to_mha + to_lora + x
        normed = self.norm(add_together)
        mlped = self.mlp(normed)
        return mlped + add_together


class TinyGPTVBlock(nn.Module):
    """
    TinyGPTVBlock is a building block for the TinyGPTV model.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        causal (bool, optional): Whether to use causal attention. Defaults to False.
        masked (bool, optional): Whether to use masked attention. Defaults to False.
        masked_seqlen (int, optional): The length of the masked sequence. Defaults to None.
        expansion_factor (int, optional): The expansion factor for the MLP. Defaults to 4.
    """

    @enforce_types
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        causal: bool = False,
        masked: bool = False,
        masked_seqlen: Optional[int] = None,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.causal = causal
        self.masked = masked
        self.masked_seqlen = masked_seqlen
        self.dropout = dropout
        self.scale = dim**-0.5
        self.mha = MultiQueryAttention(dim, heads)
        self.mlp = MLP(
            dim_in=dim, dim_out=dim, expansion_factor=expansion_factor
        )
        self.norm = nn.LayerNorm(dim)
        self.lora = Lora(dim, dim, alpha=2)
        self.rmsnorm = RMSNorm(dim)

    @enforce_types
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TinyGPTVBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        normed_x = self.norm(x)
        lorad = self.lora(normed_x)
        attn, _, _ = self.mha(normed_x)
        attn_with_lora = attn + lorad
        rms_normed = self.rmsnorm(attn_with_lora)
        mlped_x = self.mlp(x)
        return rms_normed + mlped_x
