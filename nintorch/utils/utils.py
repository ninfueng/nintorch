import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

__all__ = ['AvgMeter', 'time_loader', 'get_gpu_usage']


class AvgMeter:
    """Computes and stores the average and current value

    From: https://github.com/pytorch/examples/blob/main/imagenet/main.py

    Example:

    >>> top1 = AvgMeter()
    >>> acc1, acc5 = accuracy(output, target, topk=(1, 5))
    >>> top1.update(acc1[0], images.size(0))
    >>> top1.all_reduce()
    >>> top1.avg
    """

    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Any, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self) -> None:
        # Requires to be CUDA Tensor, otherwise not work in IBM server.
        # This may cause errors (time-out error) if not use torch.cuda.set_device() first.
        total = torch.cuda.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f'Sum: {self.sum}, Count: {self.count}, Avg: {self.avg}'


def time_loader(
    data_loader: DataLoader,
    num_workers_to_test: tuple[int, ...] = tuple(range(1, os.cpu_count())),
    num_test_epochs: int = 10,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True,
) -> int:
    """Given a `data_loader`, return an optimized number of workers with minimize
    load-times.
    """

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
            logger.info(f'Number of workers: {num_worker}, using time: {timing}.')
        timings.append(timing)

    best_timing_idx = np.argmin(timings)
    best_num_workers = num_workers_to_test[best_timing_idx]
    return best_num_workers


def get_gpu_usage(device: Optional[torch.device] = None) -> int:
    """Get a GPU usage in MB.

    Examples:

    >>> device = torch.device('cuda:0')
    >>> mem = get_gpu_usage(device)
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    mem //= 1024 * 1024
    torch.cuda.reset_peak_memory_stats()
    return mem
