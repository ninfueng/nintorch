import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__file__)
DETERMINISTIC = False

__all__ = ['seed_torch', 'disable_debug', 'enable_tf32', 'set_benchmark']


def seed_torch(seed: int, verbose: bool = False) -> None:
    """Set random seed by utilizing this function will set `DETERMINISTIC` to True."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    global DETERMINISTIC
    DETERMINISTIC = True
    if verbose:
        logger.info(f'Set a random seed: {seed} and set a `DETERMINISTIC` to True.')


# https://github.com/Lightning-AI/lightning/issues/3484
def disable_debug(verbose: bool = False) -> None:
    """Disable all `torch` debugging APIs to decrease the runtime."""
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.set_detect_anomaly(mode=False)
    if verbose:
        logger.info('Disable all `torch` debugging APIs for a faster runtime.')


def enable_tf32(verbose: bool = False) -> None:
    """Utilizes Nvidia TensorFormat 32 (TF32) datatype if detects the Ampere architecture or newer."""
    major_version, _ = torch.cuda.get_device_capability()
    if major_version >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if verbose:
            logger.info(
                'Detect an `Ampere` or a newer GPU architecture with `torch` > 1.7.0. Enable NVIDIA `TF32` datatype.'
            )


def set_benchmark(verbose: bool = False) -> None:
    global DETERMINISTIC
    if not DETERMINISTIC:
        torch.backends.cudnn.benchmark = True
        if verbose:
            logger.info('Set `torch.backends.cudnn.benchmark` to True.')
    else:
        torch.backends.cudnn.benchmark = False
        if verbose:
            logger.info('Detect `DETERMINISTIC`, set `torch.backends.cudnn.benchmark` to False.')
