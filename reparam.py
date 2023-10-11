import torch.nn.functional as F
from torch import nn


class ReparamLinear(nn.Linear):
    """
    Not Works:
    * `F.sigmoid`

    Works:
    * `F.tanh`
    * `F.hardtanh`
    * `F.softsign`
    * `torch.arctan`
    """

    def __init__(self, in_features, out_features, bias, reparam_fn=F.hardtanh, alpha=10.0) -> None:
        super().__init__(in_features, out_features, bias)
        self.reparam_fn = reparam_fn
        self.alpha = alpha

    def forward(self, input):
        weight = self.reparam_fn(self.weight) * self.alpha
        output = F.linear(input, weight, self.bias)
        return output


class ReparamConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        reparam_fn=F.hardtanh,
        alpha=10.0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.reparam_fn = reparam_fn
        self.alpha = alpha

    def forward(self, input):
        weight = self.reparam_fn(self.weight) * self.alpha
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
