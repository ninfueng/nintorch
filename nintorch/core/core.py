import numpy as np
import torch
from torch import Tensor

__all__ = [
    "print_stat",
    "torch_np",
    "np_torch",
]


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
        f"#inf: {a.isinf().sum()}, #nonzeros: {a.nonzero()}\n"
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
        raise ValueError(f"Not supporting with shape of `{len(x.shape)}`.")
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
        raise ValueError(f"Not supporting with shape of {len(shape)}.")
    return x
