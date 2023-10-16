"""Reparameterization weight layers:

Not Works: `F.sigmoid`

Works:
* `F.tanh`
* `F.hardtanh`
* `F.softsign`
* `torch.arctan`
"""
from typing import Callable

import torch.nn.functional as F
from torch import Tensor, nn


class ReparamLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        reparam_fn: Callable[..., Tensor] = F.gelu,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.reparam_fn = reparam_fn

    def forward(self, input: Tensor) -> Tensor:
        weight = self.reparam_fn(self.weight)
        output = F.linear(input, weight, self.bias)
        return output


class ReparamConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        reparam_fn: Callable[..., Callable] = F.gelu,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.reparam_fn = reparam_fn

    def forward(self, input):
        weight = self.reparam_fn(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
