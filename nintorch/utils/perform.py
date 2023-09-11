"""Performance related functions."""
import logging
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__file__)
DETERMINISTIC_FLAG = False

__all__ = [
    "set_random_seed",
    "disable_debug",
    "enable_tf32",
    "enable_perform",
    "timing_loader",
]


def set_random_seed(seed: int, verbose: bool = True) -> None:
    """Set random seed by utilizing this function will set `DETERMINISTIC_FLAG` to True."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global DETERMINISTIC_FLAG
    DETERMINISTIC_FLAG = True

    if verbose:
        logger.info(f"Set a random seed: {seed} with a DETERMINISTIC mode.")


def disable_debug(verbose: bool = True) -> None:
    """Disable all `torch` debugging APIs to decrease runtime."""
    # https://github.com/Lightning-AI/lightning/issues/3484
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.set_detect_anomaly(mode=False)

    if verbose:
        logger.info("Disable all `torch` debugging APIs for a faster runtime.")


def enable_tf32(verbose: bool = True) -> None:
    """Utilizes Nvidia TensorFormat 32 (TF32) datatype if detects the Ampere architecture or newer."""

    major_version, _ = torch.cuda.get_device_capability()
    if major_version >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            logger.info(
                "Detect an `Ampere` or a newer GPU architecture with `torch` > 1.7.0. "
                "Enable NVIDIA `TF32` datatype."
            )


def enable_perform(verbose: bool = True) -> None:
    """Try to use `TF32` datatype and disable debug mode if `DETERMINISTIC_FLAG == False`
    Assign `torch.backends.cudnn.benchmark = True`."""

    global DETERMINISTIC_FLAG
    enable_tf32(verbose=verbose)
    disable_debug(verbose=verbose)

    if not DETERMINISTIC_FLAG:
        torch.backends.cudnn.benchmark = True
        if verbose:
            logger.info(
                "Detect `DETERMINISTIC_FLAG`, "
                "set `torch.backends.cudnn.benchmark = True`."
            )


def timing_loader(
    data_loader: DataLoader,
    num_workers_to_test: Tuple[int, ...] = tuple(range(1, os.cpu_count())),
    num_test_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> int:
    """Given a `data_loader`, return an optimized number of workers with minimize load-times."""

    timings = []
    for num_worker in num_workers_to_test:
        data_loader.num_workers = num_worker
        t0 = time.perf_counter()

        for _ in range(num_test_epochs):
            for data, label in data_loader:
                data, label = data.to(device), label.to(device)

        t1 = time.perf_counter()
        timing = t1 - t0
        if verbose:
            logger.info(f"Number of workers: {num_worker}, using time: {timing}.")
        timings.append(timing)

    best_timing_idx = np.argmin(timings)
    best_num_workers = num_workers_to_test[best_timing_idx]
    return best_num_workers
