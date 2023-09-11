from functools import reduce

import numpy as np
import torch
from torch import Tensor, nn


__all__ = [
    "get_numel_model",
    "torch_np",
    "np_torch",
    "count_model_size",
    "print_stat",
]


def get_numel_model(
    module: nn.Module, count_requires_grad: bool = False, count_nonzero: bool = False
) -> int:
    """Return a number of parameters with `require_grad=True` of `nn.Module`."""
    if count_nonzero:
        all_params = [
            param.count_nonzero().detach().numpy()
            for param in module.parameters()
            if not count_requires_grad or (count_requires_grad and param.requires_grad)
        ]
    else:
        all_params = [
            param.numel()
            for param in module.parameters()
            if not count_requires_grad or (count_requires_grad and param.requires_grad)
        ]
    numel = reduce(lambda x, y: x + y, all_params)
    return numel


def print_stat(a: Tensor) -> None:
    """Print `Tensor` with statistical information.

    Arguments:
        a: a tensor to print statistical information with.

    Returns:
        None
    """
    print(
        f"shape: {a.shape}\n"
        f"numel: {a.numel()}\n"
        f"range: [{a.amin():.6f}, {a.amax():.6f}]\n"
        f"μ: {a.mean():.6f}, σ: {a.std():.6f}\n"
        f"#inf: {a.isinf().sum()}, #zeros: {(a == 0).sum()}\n"
    )


def torch_np(x: Tensor) -> np.ndarray:
    """Convert from `Tensor` NCHW to `np.ndarray` NHWC format."""
    assert isinstance(x, Tensor)
    x = x.detach().cpu()
    if len(x.shape) == 2:
        x = torch.movedim(x, 1, 0)
    elif len(x.shape) == 3:
        x = torch.movedim(x, 0, 2)
    elif len(x.shape) == 4:
        x = x.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Not supporting with shape of `{len(x.shape)}`.")
    return x.numpy()


def np_torch(x: np.ndarray) -> Tensor:
    """Convert from NHWC `np.ndarray` to NCHW `Tensor` format."""
    shape = x.shape
    x = torch.from_numpy(x)
    if len(shape) == 2:
        x = torch.movedim(x, -1, 0)
    elif len(shape) == 3:
        x = torch.movedim(x, -1, 0)
    elif len(shape) == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Not supporting with shape of {len(shape)}.")
    return x


def count_model_size(
    model: nn.Module, bits: int = 32, count_nonzero: bool = False
) -> str:
    """Count model size byte term and return with a suitable unit upto `GiB`."""
    Byte = 8
    KB = 1_024 * Byte
    MB = 1_024 * KB
    GB = 1_024 * MB

    numel = get_numel_model(model, count_nonzero)
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
