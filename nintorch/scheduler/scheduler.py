from typing import List, Optional, Union

import torch.nn as nn
import torch.optim as optim
from nincore import to_1tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR, StepLR

__all__ = ['WarmupLR', 'insert_warmup']


class WarmupLR(LRScheduler):
    """Optimizer increments `self.last_epoch` which starts from -1.

    Arguments:
        max_iterations: a number of iterations to reach the maximum lr

    Example:
    >>> model = nn.Linear(10, 1)
    >>> optimizer = optim.Adam(model.parameters(), lr=1.0)
    >>> scheduler = WarmupLR(optimizer, max_iterations=3)
    >>> lrs = []
    >>> for i in range(5):
    >>>     optimizer.step()
    >>>     scheduler.step()
    >>>     lrs.append(scheduler.get_lr()[0])
    >>> lrs
    [0.3333333333333333, 0.6666666666666666, 1.0, 1.0, 1.0]
    >>> scheduler.done
    >>> True
    """

    def __init__(self, optimizer: Optimizer, max_iterations: int, max_lr: Optional[float] = None) -> None:
        self.max_lr = max_lr if max_lr is not None else optimizer.param_groups[0]['lr']
        self.max_iterations = max_iterations
        self.done = False
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.max_iterations:
            groups = []
            for _ in self.optimizer.param_groups:
                lr_group = self.max_lr * (self.last_epoch / self.max_iterations)
                groups.append(lr_group)
            return groups
        else:
            self.done = True
            return [group['lr'] for group in self.optimizer.param_groups]


# TODO: testing this functions
def insert_warmup(
    max_iterations: int,
    optimizer: Optimizer,
    schedulers: Union[LRScheduler, List[LRScheduler]],
    milestones: List[int],
) -> LRScheduler:
    """Insert `WarmupLR` and `schedulers` to `SequentialLR`."""

    max_lr = optimizer.param_groups[0]['lr']
    schedulers = list(to_1tuple(schedulers))
    assert len(schedulers) == len(milestones) - 1, (
        'Length of schedulers should be equal length of milestones minus one. '
        f'Your {len(schedulers)=} != {len(milestones)}.'
    )

    warmup_scheduler = WarmupLR(optimizer, max_iterations, max_lr)
    schedulers = [warmup_scheduler] + schedulers

    scheduler_lr_with_warmup = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    return scheduler_lr_with_warmup


if __name__ == '__main__':
    from torch.optim.lr_scheduler import StepLR

    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1.0)
    scheduler = WarmupLR(optimizer, max_iterations=4)
    scheduler2 = StepLR(optimizer, 8)

    lrs = []
    for i in range(10):
        optimizer.step()
        scheduler.step()
        scheduler2.step()
        print(scheduler.done)
        lrs.append(scheduler.get_lr()[0])
    print(lrs)
