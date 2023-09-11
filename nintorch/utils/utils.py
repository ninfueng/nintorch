"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import glob
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_logger(
    log_dir: Union[str, Path],
    base_level: int = logging.INFO,
    to_stdout: bool = True,
    rm_exist: bool = True,
    with_color: bool = True,
) -> None:
    """Set the logger to log info in terminal and file `log_dir`.
    In general, it is useful to have a logger so that every output to the terminal
    is saved in a permanent file. Here we save it to `model_dir/train.log`.

    Args:
        log_dir: (string) location of log file
        log_level: (string) set log level
        to_stdout: (bool) whether to print log to stdout
        rm_exist: (bool) remove the old log file before start log or not
        verbose: (bool) if True, verbose some information

    Example:
    >>> set_logger("info.log")
    >>> logger.info("Starting training...")
    """
    assert isinstance(to_stdout, bool)
    assert base_level in [0, 10, 20, 30, 40, 50]
    assert isinstance(rm_exist, bool)

    log_dir = Path(log_dir)
    log_dir = log_dir.expanduser()
    parent_dir = log_dir.parent

    if not parent_dir.is_dir():
        parent_dir.mkdir(exist_ok=True)

    if rm_exist and parent_dir.is_file():
        log_dir.unlink()

    logger = logging.getLogger()
    logger.setLevel(base_level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s:%(filename)s: %(message)s")
        )
        logger.addHandler(file_handler)

        if to_stdout:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s:%(filename)s: %(message)s")
            )
            logger.addHandler(stream_handler)

    if with_color:
        # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
        if sys.stderr.isatty():
            logging.addLevelName(
                logging.INFO, f"\033[1;32m{logging.getLevelName(logging.INFO)}\033[1;0m"
            )
            logging.addLevelName(
                logging.WARNING,
                f"\033[1;33m{logging.getLevelName(logging.WARNING)}\033[1;0m",
            )
            logging.addLevelName(
                logging.ERROR,
                f"\033[1;31m{logging.getLevelName(logging.ERROR)}\033[1;0m",
            )
            logging.addLevelName(
                logging.CRITICAL,
                f"\041[1;31m{logging.getLevelName(logging.ERROR)}\033[1;0m",
            )


def backup_scripts(file_type: Union[str, Sequence], destination: str) -> None:
    """Copy all files with `file_type` to the destination location."""
    scripts = []
    os.makedirs(destination, exist_ok=True)

    if isinstance(file_type, str):
        scripts += glob.glob(os.path.join(os.curdir, f"*{file_type}"))
    elif isinstance(file_type, Sequence):
        for f in file_type:
            scripts += glob.glob(os.path.join(os.curdir, f"*{f}"))
    else:
        raise NotImplementedError()

    for s in scripts:
        file_name = os.path.basename(s)
        shutil.copy2(os.path.join(s), os.path.join(destination, file_name))


class AvgMeter(object):
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
        return f"Sum: {self.sum}, Count: {self.count}, Avg: {self.avg}"


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    From: https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(1 / batch_size))
        return results


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py
def add_weight_decay(
    model: nn.Module,
    weight_decay: float = 1e-5,
    no_weight_decay_list: Union[Tuple[str], Tuple[()]] = (),
) -> List[Dict[str, Any]]:
    """Add weight decay if `ndim == 1`, in `no_weight_decay_list`, or name end with `.bias`.

    Example:
    >>> from torchvision.models import resnet18
    >>> model = resnet18()
    >>> named_params = add_weight_decay(model, 1e-3)
    """
    no_weight_decay_list = set(no_weight_decay_list)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # `param.ndim <= 1` will filter-out BatchNorm1d, 2d, and 3d to `no_decay`.
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
