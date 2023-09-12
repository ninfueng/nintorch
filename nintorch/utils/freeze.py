import logging
from typing import List, Union

import torch.nn as nn
from nincore import multi_getattr, to_1tuple

logger = logging.getLogger("__file__")

try:
    from torch.nn.modules.batchnorm import _NormBase

    BaseNorm = _NormBase

except ImportError:
    from torch.nn.modules.batchnorm import _BatchNorm

    BaseNorm = _BatchNorm


__all__ = ["freeze_norm", "freeze_except"]


def freeze_norm(model: nn.Module, verbose: bool = False) -> None:
    """Disable statistic tracking and training of Norm layers like: `InstanceNorm` or `BatchNorm`.

    Arguments:
        model: a model to freeze the batch normalization
        verbose: whether to logging name of freeze modules or not.
    Returns:
        None

    Examples:
    >>> model = resnet18()
    >>> freeze_norm(model)
    """
    for n, m in model.named_modules():
        if isinstance(m, BaseNorm):
            m.track_running_stats = False
            if hasattr(m, "weight"):
                m.weight.requires_grad_(False)
            elif hasattr(m, "bias"):
                m.bias.requires_grad_(False)
            if verbose:
                logger.info(f"`{n}` is freeze with `freeze_norm`.")


def freeze_except(model: nn.Module, except_names: Union[List[str], str]) -> nn.Module:
    except_names = to_1tuple(except_names)
    model.requires_grad_(False)
    for e in except_names:
        multi_getattr(model, e).requires_grad_(True)
    return model
