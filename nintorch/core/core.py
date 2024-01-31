from typing import List, Optional, Tuple
from functools import reduce

import numpy as np
import torch
from torch import Tensor


__all__ = [
    'print_stat',
    'torch_np',
    'np_torch',
    'torch_choice',
]


def print_stat(a: Tensor) -> None:
    """Print `Tensor` with statistical information.

    Arguments:
        a: a tensor to print statistical information with.
    """
    print(
        f'shape: {a.shape}, numel: {a.numel()}\n'
        f'range: [{a.amin():.6f}, {a.amax():.6f}]\n'
        f'μ: {a.mean():.6f}, σ: {a.std():.6f}\n'
        f'#inf: {a.isinf().sum()}, #nonzeros: {a.nonzero()}\n'
    )


def torch_np(x: Tensor) -> np.ndarray:
    """Convert from `Tensor` NCHW to `np.ndarray` NHWC format."""
    assert isinstance(x, Tensor)
    x = x.detach().cpu()
    ndim = x.ndim
    if ndim == 2:
        x = torch.movedim(x, 1, 0)
    elif ndim == 3:
        x = torch.movedim(x, 0, 2)
    elif ndim == 4:
        x = x.permute(0, 2, 3, 1)
    else:
        raise ValueError(f'Not supporting with shape of `{len(x.shape)}`.')
    return x.numpy()


def np_torch(x: np.ndarray) -> Tensor:
    """Convert from NHWC `np.ndarray` to NCHW `Tensor` format."""
    ndim = x.ndim
    x = torch.from_numpy(x)
    if ndim == 2:
        x = torch.movedim(x, -1, 0)
    elif ndim == 3:
        x = torch.movedim(x, -1, 0)
    elif ndim == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f'Not supporting with shape of {len(shape)}.')
    return x


def torch_choice(
    choice: List[int],
    shape: Tuple[int, ...],
    p: Optional[float] = None,
    device: torch.device = torch.device('cpu'),
) -> Tensor:
    """
    >>> choice = torch.tensor([1, 2, 3])
    >>> shape = (3, 4)
    >>> p = [0.3_333, 0.3_333, 0.3_333]
    >>> device = torch.device('cpu')
    >>> choice = torch_choice(choice, shape, p, device)
    """
    choice = torch.as_tensor(choice, device=device)
    if p is None:
        p = torch.ones_like(choice) / len(choice)
    else:
        p = torch.as_tensor(p, device=device)
    size = reduce(lambda x, y: x * y, shape)
    idx = p.multinomial(size, replacement=True)
    choice = choice[idx]
    choice = choice.reshape(shape)
    return choice
