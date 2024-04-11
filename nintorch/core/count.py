import time
from functools import reduce
from typing import Sequence

import torch
from nincore.utils import AvgMeter
from torch import Tensor, nn
from torch.nn.utils import prune

from nintorch.core import infer_mode

TORCH_PROFILE = False
try:
    from torchprofile import profile_macs

    TORCH_PROFILE = True
except ImportError:
    raise ImportError(
        '`count_macs` requires `torchprofile`.'
        'Please install via `pip install torchprofile`.'
    )


__all__ = [
    'count_params',
    'count_size',
    'count_sparse',
    'count_sparse_module',
    'count_latency',
]


@infer_mode()
def count_params(
    model: nn.Module,
    only_requires_grad: bool = False,
    only_nonzero: bool = False,
    bias: bool = True,
    return_layers: bool = False,
) -> int | dict[str, int]:
    """Return a number of parameters with `require_grad=True` of `nn.Module`.

    Default = counting all parameters and with both zero and nonzero.

    Args:
        model: a model to be count
        only_requires_grad: count only parameters with require_grad=True
        only_nonzero: count only nonzero parameters
        return_layers: return a dict with layer by layer parameters.
    """
    # If `only_requires_grad = False`, all parameters counted.
    # If `only_requires_grad = True`, only `required_grad=True` counted.
    all_params, all_params_dict = [], {}
    for name, param in model.named_parameters():
        if not bias and name.find('bias') > -1:
            continue

        if not only_requires_grad or param.requires_grad:
            if only_nonzero:
                nonzero = param.count_nonzero().detach().numpy()
                if not return_layers:
                    all_params.append(nonzero)
                else:
                    all_params_dict.update({name: nonzero})
            else:
                numel = param.numel()
                if not return_layers:
                    all_params.append(numel)
                else:
                    all_params_dict.update({name: numel})

    if not return_layers:
        numel = reduce(lambda x, y: x + y, all_params)
        return numel
    else:
        return all_params_dict


@infer_mode()
def count_size(
    model: nn.Module,
    bits: int = 32,
    only_requires_grad: bool = False,
    only_nonzero: bool = False,
) -> str:
    """Count model size byte term and return with a suitable unit upto `GB`."""
    Byte = 8
    KB = 1_024 * Byte
    MB = 1_024 * KB
    GB = 1_024 * MB

    numel = count_params(
        model,
        only_requires_grad=only_requires_grad,
        only_nonzero=only_nonzero,
        return_layers=False,
    )
    size = numel * bits
    unit = 'B'
    if size >= GB:
        size = size / GB
        unit = 'GB'
    elif size >= MB:
        size = size / MB
        unit = 'MB'
    elif size >= KB:
        size = size / KB
        unit = 'KB'
    return f'{size:4f} {unit}'


@infer_mode()
def count_sparse(param: Tensor) -> float:
    """"""
    numel = param.numel()
    num_zero = numel - param.count_nonzero()
    sparse = num_zero / numel
    return sparse.item()


@infer_mode()
def count_sparse_module(
    model: nn.Module,
    bias: bool = True,
    return_layers: bool = False,
) -> float | dict[str, float]:
    """Measure sparsity given `nn.Module`.

    If able to detect `torch.nn.utils.prune` will looks for `model.named_buffers`
    instead of `model.named_parameters`.
    """
    is_pruned = prune.is_pruned(model)
    if is_pruned:
        named_paramaters = model.named_buffers()
    else:
        named_paramaters = model.named_parameters()

    all_sparse_dict = {}
    sparses = AvgMeter()
    for name, param in named_paramaters:
        if not bias and name.find('bias') > -1:
            continue

        sparse = count_sparse(param)
        numel = param.numel()
        if return_layers:
            all_sparse_dict.update({name: sparse})
        else:
            sparses.update(sparse, numel)

    if return_layers:
        return all_sparse_dict
    else:
        return sparses.avg


@infer_mode()
def count_latency(
    model: nn.Module, dummy_input: Tensor, n_warmup: int = 20, n_test: int = 100
) -> float:
    """Measure average latency of any `nn.Module` with a given `dummy_input`.

    1. Run a `model` with `n_warmup` rounds first.
    2. Run a `model` with `n_test` rounds and record the running times.
    3. Return an average latency.
    """
    model.eval()
    for _ in range(n_warmup):
        _ = model(dummy_input)

    t1 = time.perf_counter()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.perf_counter()
    return (t2 - t1) / n_test


# While can using with `torchprofile.count_mac` directly, but fn is just for a reminder.
if TORCH_PROFILE:
    __all__.append('count_macs')

    @infer_mode()
    def count_macs(
        model: nn.Module,
        input_size: Sequence[int],
        device: torch.device = torch.device('cuda'),
    ) -> int:
        model = model.to(device)
        input = torch.empty(input_size, device=device)
        macs = profile_macs(model, input)
        return macs
