import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from quant import BinQuant, TerQuant, get_thres


class EmbedBinLinear(nn.Linear):
    """Embedded Binarized Linear layer.

    Attributes:
        forward: forward with floating point weights.
        forward_bin: forward with binary weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.quant_weight = None

    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.bias)
        return output

    def forward_bin(self, input: Tensor) -> Tensor:
        self.quant_weight = BinQuant.apply(self.weight)
        quant_weight = self.quant_weight
        output = F.linear(input, quant_weight, self.bias)
        return output

    def get_quant_weight(self) -> Tensor:
        if self.quant_weight is None:
            return BinQuant.apply(self.weight)
        return self.quant_weight


class EmbedBinConv2d(nn.Conv2d):
    """Embedded Binarized Conv2d layer.

    Attributes:
        forward: forward with floating point weights.
        forward_bin: forward with binary weights.
    """

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
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_weight = None

    def forward(self, input: Tensor) -> Tensor:
        output = F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def forward_bin(self, input: Tensor) -> Tensor:
        self.quant_weight = BinQuant.apply(self.weight)
        quant_weight = self.quant_weight
        output = F.conv2d(
            input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def get_quant_weight(self) -> Tensor:
        if self.quant_weight is None:
            return BinQuant.apply(self.weight)
        return self.quant_weight


class EmbedBinTerLinear(nn.Linear):
    """Embedded Binarized Linear layer.

    Attributes:
        forward: forward with floating point weights.
        forward_bin: forward with binary weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.quant_weight = None

    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.bias)
        return output

    def forward_bin(self, input: Tensor) -> Tensor:
        self.quant_weight = BinQuant.apply(self.weight)
        quant_weight = self.quant_weight
        output = F.linear(input, quant_weight, self.bias)
        return output

    def forward_ter(self, input: Tensor) -> Tensor:
        thres = get_thres(self.weight)
        self.quant_weight = TerQuant.apply(self.weight, thres)
        quant_weight = self.quant_weight
        output = F.linear(input, quant_weight, self.bias)
        return output

    def get_quant_weight(self) -> Tensor:
        if self.quant_weight is None:
            return BinQuant.apply(self.weight)
        return self.quant_weight


class EmbedBinTerConv2d(nn.Conv2d):
    """Embedded Binarized Conv2d layer.

    Attributes:
        forward: forward with floating point weights.
        forward_bin: forward with binary weights.
    """

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
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_weight = None

    def forward(self, input: Tensor) -> Tensor:
        output = F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def forward_bin(self, input: Tensor) -> Tensor:
        self.quant_weight = BinQuant.apply(self.weight)
        quant_weight = self.quant_weight
        output = F.conv2d(
            input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def forward_ter(self, input: Tensor) -> Tensor:
        thres = get_thres(self.weight)
        self.quant_weight = TerQuant.apply(self.weight, thres)
        quant_weight = self.quant_weight
        output = F.conv2d(
            input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def get_bin_weight(self) -> Tensor:
        if self.quant_weight is None:
            return BinQuant.apply(self.weight)
        return self.quant_weight

    def get_ter_weight(self) -> Tensor:
        thres = get_thres(self.weight)
        if self.quant_weight is None:
            return TerQuant.apply(self.weight, thres)
        return self.quant_weight
