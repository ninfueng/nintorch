from functools import partial
from typing import Callable, List

import torch.nn as nn
from torch import Tensor

__all__ = ['PackedNorm', 'PackedBatchNorm1d', 'PackedBatchNorm2d']


class PackedNorm(nn.Module):
    def __init__(
        self,
        norm_layer: Callable[..., nn.Module],
        num_bns: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.bns = nn.Sequential(
            *[
                norm_layer(
                    num_features,
                    eps,
                    momentum,
                    affine,
                    track_running_stats,
                    *args,
                    **kwargs,
                )
                for _ in range(num_bns)
            ]
        )

    def forward(self, input: Tensor) -> List[Tensor]:
        return [bn(input) for bn in self.bns]

    def forward_idx(self, input: Tensor, idx: int) -> Tensor:
        assert idx < len(self.bns) - 1, f'`idx` out of range, `idx` < {len(self.bns) - 1}.'
        bn = self.bns[idx]
        output = bn(input)
        return output


PackedBatchNorm1d = partial(PackedNorm, nn.BatchNorm1d)
PackedBatchNorm2d = partial(PackedNorm, nn.BatchNorm2d)


if __name__ == '__main__':
    import torch

    pack_norm = PackedBatchNorm1d(3, 10)
    input = torch.randn(2, 10, 10)
    output = pack_norm.forward_idx(input, 0)
    print(output.shape)
