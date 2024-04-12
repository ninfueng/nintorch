from functools import reduce

import numpy as np
import torch
from torch import Tensor

__all__ = [
    'print_stat',
    'torch_np',
    'np_torch',
    'torch_choice',
    'get_device',
]


@torch.inference_mode()
def print_stat(a: Tensor | np.ndarray) -> None:
    """Print `Tensor` with statistical information:

    - Shape
    - #Numel
    - #Inf
    - #NaN
    - #Zero
    - Range
    - Mean ± Std

    Arguments:
        a: a tensor to print statistical information with.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)

    numel = a.numel()
    n_inf = a.isinf().sum()
    n_nan = a.isnan().sum()
    n_zero = numel - a.count_nonzero().item()
    print(
        f'Shape : {tuple(a.shape)}\n'
        f'#Numel: {numel:,}\n'
        f'#Inf  : {n_inf:,} ({n_inf/numel:.6f}%)\n'
        f'#NaN  : {n_nan:,} ({n_nan/numel:.6f}%)\n'
        f'#Zero : {n_zero:,} ({n_zero/numel:.6f}%)\n'
        f'⊆ [{a.amin().item():,.6f}, {a.amax().item():,.6f}]\n'
        f'{a.mean().item():,.6f} ± {a.std().item():,.6f}'
    )


@torch.inference_mode()
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
    choice: list[int],
    shape: tuple[int, ...],
    p: list[float] | None = None,
    device: torch.device = torch.device('cpu'),
) -> Tensor:
    """torch version of `np.random.choice`.

    Example:
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


def get_device(device_id: int | None = None) -> torch.device:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    device = f'{device}:{device_id}' if device_id is not None else device
    return torch.device(device)


if __name__ == '__main__':
    input = torch.rand(1, 2, 3, 4)
    print_stat(input)
