import logging

import torch.nn as nn

logger = logging.getLogger("__file__")

try:
    from torch.nn.modules.batchnorm import _NormBase

    BaseNorm = _NormBase

except ImportError:
    from torch.nn.modules.batchnorm import _BatchNorm

    BaseNorm = _BatchNorm
    logger.warn(
        "Cannot import `torch.nn.modules.batchnorm._NormBase`."
        "Import `torch.nn.modules.batchnorm._BatchNorm`."
    )


def freeze_norm(model: nn.Module) -> None:
    """Freeze all tracking statistics and training of all parameters
    in `InstanceNorm` or `BatchNorm`.

    Arguments:
        model: a model to freeze the batch normalization.

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

            logger.info(f"`{n}` is freeze with `freeze_norm`.")
