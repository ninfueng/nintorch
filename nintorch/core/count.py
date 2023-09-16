from functools import reduce
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = ["count_params", "count_macs", "count_size", "count_sparse"]


def count_params(
    model: nn.Module,
    count_only_requires_grad: bool = False,
    count_only_nonzero: bool = False,
) -> int:
    """Return a number of parameters with `require_grad=True` of `nn.Module`.

    Default = counting all parameters and with both zero and nonzero.

    Args:
        model: a model to be count
        count_only_requires_grad: count only parameters with require_grad=True
        count_only_nonzero: count only nonzero parameters
    """
    if count_only_nonzero:
        # Tensor supports a `nonzero()` attribute.
        all_params = [
            param.count_nonzero().detach().numpy()
            for param in model.parameters()
            # If `count_only_requires_grad = False`, all parameters counted.
            # If `count_only_requires_grad = True`, only `required_grad=True` counted.
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
    count_only_nonzero: bool = False,
) -> str:
    """Count model size byte term and return with a suitable unit upto `GB`."""
    Byte = 8
    KB = 1_024 * Byte
    MB = 1_024 * KB
    GB = 1_024 * MB

    numel = count_params(
        model,
        count_only_requires_grad=count_only_requires_grad,
        count_only_nonzero=count_only_nonzero,
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


def count_sparse(model: nn.Module, skip_bias: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Measure sparsity given `nn.Module`.

    If able to detect `torch.nn.utils.prune` will looks for `model.named_buffers`
    instead of `model.named_parameters`.
    """
    # TODO: maybe using `AvgMeter` instead?
    name_sparse, numels, num_zeros = {}, 0, 0
    is_pruned = nn.utils.prune.is_pruned(model)
    if is_pruned:
        named_params = model.named_buffers()
    else:
        named_params = model.named_parameters()

    for name, param in named_params:
        if skip_bias and name.find("bias") > -1:
            continue
        zero_mask = torch.where(param == 0.0, torch.ones_like(param), torch.zeros_like(param))
        num_zero, numel = zero_mask.sum(), param.numel()
        sparse = num_zero / numel

        numels += numel
        num_zeros += num_zero
        name_sparse.update({name: sparse})

    sparses = num_zeros / numels
    return sparses, name_sparse
