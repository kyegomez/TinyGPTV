import torch
from torch import nn
from tiny_gptv.blocks import SkipThenAdd


def test_skip_then_add():
    # Create a SkipThenAdd instance with two modules
    module1 = nn.Linear(512, 512)
    module2 = nn.ReLU()
    skip_then_add = SkipThenAdd(modules=[module1, module2])

    # Create a random tensor of size (batch_size, sequence_length, dim)
    x = torch.randn(1, 10, 512)

    # Forward pass
    output = skip_then_add(x)

    # Check output shape
    assert output.shape == x.shape

    # Check that output is not equal to input (i.e., transformation has occurred)
    assert not torch.all(torch.eq(output, x))


def test_skip_then_add_with_args():
    # Create a SkipThenAdd instance with two modules that take additional arguments
    module1 = nn.Linear(512, 512)
    module2 = nn.ReLU()
    skip_then_add = SkipThenAdd(
        modules=[module1, module2], arg1=1, arg2="abc"
    )

    # Create a random tensor of size (batch_size, sequence_length, dim)
    x = torch.randn(1, 10, 512)

    # Forward pass
    output = skip_then_add(x, arg3=True)

    # Check output shape
    assert output.shape == x.shape

    # Check that output is not equal to input (i.e., transformation has occurred)
    assert not torch.all(torch.eq(output, x))


def test_skip_then_add_multiple_modules():
    # Create a SkipThenAdd instance with three modules
    module1 = nn.Linear(512, 512)
    module2 = nn.ReLU()
    module3 = nn.BatchNorm1d(512)
    skip_then_add = SkipThenAdd(modules=[module1, module2, module3])

    # Create a random tensor of size (batch_size, sequence_length, dim)
    x = torch.randn(1, 10, 512)

    # Forward pass
    output = skip_then_add(x)

    # Check output shape
    assert output.shape == x.shape

    # Check that output is not equal to input (i.e., transformation has occurred)
    assert not torch.all(torch.eq(output, x))


def test_skip_then_add_empty_modules():
    # Create a SkipThenAdd instance with no modules
    skip_then_add = SkipThenAdd()

    # Create a random tensor of size (batch_size, sequence_length, dim)
    x = torch.randn(1, 10, 512)

    # Forward pass
    output = skip_then_add(x)

    # Check output shape
    assert output.shape == x.shape

    # Check that output is equal to input (no transformation has occurred)
    assert torch.all(torch.eq(output, x))
