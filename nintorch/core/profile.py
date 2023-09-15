from functools import reduce
from typing import Sequence

import torch
from torch import nn

__all__ = ["count_params", "count_macs", "count_size"]


def count_params(
    model: nn.Module,
    count_only_requires_grad: bool = False,
    count_nonzero: bool = False,
) -> int:
    """Return a number of parameters with `require_grad=True` of `nn.Module`.

    Default = counting all parameters and with both zero and nonzero.

    Args:
        model: a model to be count.
        count_requires_grad: count only parameters with require_grad=True.
        count_nonzero: count only nonzero parameters.
    """
    if count_nonzero:
        # Tensor supports a `nonzero()` attribute.
        all_params = [
            param.count_nonzero().detach().numpy()
            for param in model.parameters()
            # If count_only_requires_grad = False, all parameters counted.
            # If count_only_requires_grad = True, only required_grad=True counted.
            if not count_only_requires_grad or param.requires_grad
        ]
    else:
        all_params = [
            param.numel() for param in model.parameters() if not count_only_requires_grad or param.requires_grad
        ]
    numel = reduce(lambda x, y: x + y, all_params)
    return numel


def count_size(
    model: nn.Module,
    bits: int = 32,
    count_only_requires_grad: bool = False,
    count_nonzero: bool = False,
) -> str:
    """Count model size byte term and return with a suitable unit upto `GiB`."""
    Byte = 8
    KB = 1_024 * Byte
    MB = 1_024 * KB
    GB = 1_024 * MB

    numel = count_params(
        model,
        count_only_requires_grad=count_only_requires_grad,
        count_nonzero=count_nonzero,
    )
    size, unit = numel * bits, "B"

    if size >= GB:
        size = size / GB
        unit = "GB"
    elif size >= MB:
        size = size / MB
        unit = "MB"
    elif size >= KB:
        size = size / KB
        unit = "KB"
    return f"{size:4f} {unit}"


# While can using with `torchprofile.count_mac` directly, but fn is just for a reminder.
def count_macs(
    model: nn.Module,
    input_size: Sequence[int],
    device: torch.device = torch.device("cuda"),
) -> int:
    try:
        from torchprofile import profile_macs
    except ImportError:
        raise ImportError("`count_macs` requires `torchprofile`." "Please install via `pip install torchprofile`.")

    model = model.to(device)
    input = torch.empty(input_size, device=device)
    macs = profile_macs(model, input)
    return macs
