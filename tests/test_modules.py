import torch
import pytest
from tiny_gptv.blocks import (
    LoraMHABlock,
    MultiQueryAttention,
    MLP,
    Lora,
    TinyGPTVBlock,
)


def test_multi_query_attention():
    mha = MultiQueryAttention(dim=512, heads=8)
    x = torch.randn(1, 10, 512)
    output, _, _ = mha(x)
    assert output.shape == x.shape


def test_mlp():
    mlp = MLP(dim_in=512, dim_out=512, expansion_factor=4)
    x = torch.randn(1, 10, 512)
    output = mlp(x)
    assert output.shape == x.shape


def test_lora():
    lora = Lora(dim_in=512, dim_out=512, alpha=2)
    x = torch.randn(1, 10, 512)
    output = lora(x)
    assert output.shape == x.shape


def test_lora_mha_block():
    lora_mha = LoraMHABlock(dim=512, heads=8)
    x = torch.randn(1, 10, 512)
    output = lora_mha(x)
    assert output.shape == x.shape


@pytest.mark.parametrize("dim, heads", [(512, 8), (1024, 16)])
def test_lora_mha_block_shapes(dim, heads):
    lora_mha = LoraMHABlock(dim=dim, heads=heads)
    x = torch.randn(1, 10, dim)
    output = lora_mha(x)
    assert output.shape == x.shape


def test_lora_mha_block_raises():
    with pytest.raises(ValueError):
        LoraMHABlock(dim=512, heads=10)


def test_tiny_gptv_block():
    # Initialize a TinyGPTVBlock instance
    tiny_gptv = TinyGPTVBlock(dim=512, heads=8)

    # Create a random tensor of size (batch_size, sequence_length, dim)
    x = torch.randn(1, 10, 512)

    # Forward pass
    output = tiny_gptv(x)

    # Check output shape
    assert output.shape == x.shape

    # Check that output is not equal to input (i.e., transformation has occurred)
    assert not torch.all(torch.eq(output, x))


@pytest.mark.parametrize("dim, heads", [(512, 8), (1024, 16)])
def test_tiny_gptv_block_shapes(dim, heads):
    # Initialize a TinyGPTVBlock instance
    tiny_gptv = TinyGPTVBlock(dim=dim, heads=heads)

    # Create a random tensor of size (batch_size, sequence_length, dim)
    x = torch.randn(1, 10, dim)

    # Forward pass
    output = tiny_gptv(x)

    # Check output shape
    assert output.shape == x.shape


def test_tiny_gptv_block_raises():
    # Check that TinyGPTVBlock raises an error if dim is not divisible by heads
    with pytest.raises(ValueError):
        TinyGPTVBlock(dim=512, heads=10)
