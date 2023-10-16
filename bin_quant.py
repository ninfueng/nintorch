from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['BinQuant', 'BinActQuant', 'BinConv2d', 'BinLinear']


class BinQuant(torch.autograd.Function):
    """Designed to use with weights."""

    @staticmethod
    def forward(_: Any, input: torch.Tensor, scaling: Optional[Tensor] = None) -> torch.Tensor:
        if scaling is None:
            return input.sign()
        else:
            return input.sign() * scaling

    @staticmethod
    def backward(_: Any, grad_out: torch.Tensor) -> torch.Tensor:
        """Replace backward process with do nothings instead.
        Gradients come from output -> input.
        """
        grad_input = grad_out.clone()
        return grad_input


class BinActQuant(torch.autograd.Function):
    """Basic activation quantization for binarized neural networks.
    Refer: https://github.com/itayhubara/BinaryNet.pytorch/blob/master/main_binary.py
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_input = grad_out.clone()
        # Straight through estimator = clip and identical.
        # weight is not needed to be clipped the gradient.
        # However, can be clipped afterward.
        # This equivalents to tanh.
        grad_input.masked_fill_(input > 1.0, 0.0)
        grad_input.masked_fill_(input < -1.0, 0.0)
        return grad_input


class BinLinear(nn.Linear):
    """Binarized Linear layer.

    First layer should set `quant_in_act` as False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_in_act: bool = False,
        clamp_weight: bool = False,
        scaling: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer('quant_in_act', torch.as_tensor(quant_in_act, dtype=torch.bool))
        self.register_buffer('clamp_weight', torch.as_tensor(clamp_weight, dtype=torch.bool))
        self.register_buffer('scaling', torch.as_tensor(scaling, dtype=torch.bool))
        self.register_buffer('scale', torch.as_tensor(0.0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quant_in_act:
            input = BinActQuant.apply(input)

        if self.clamp_weight:
            with torch.no_grad():
                self.weight = self.weight.clamp(-1, 1)

        quant_weight = BinQuant.apply(self.weight)
        if self.scaling:
            if self.training:
                self.scale = self.get_scale()
            quant_weight = quant_weight * self.scale

        output = F.linear(input, quant_weight, self.bias)
        return output

    def get_scale(self) -> torch.Tensor:
        return self.weight.abs().mean()

    def get_quant_weight(self) -> torch.Tensor:
        if self.scaling:
            if self.scale == 0.0:
                scale = self.get_scale()
            else:
                scale = self.scale
            return BinQuant.apply(self.weight) * scale
        else:
            return BinQuant.apply(self.weight)

    def __repr__(self) -> str:
        info = super().__repr__()[:-1] + f', quant_in_act={self.quant_in_act}' + f', clamp_weight={self.clamp_weight})'
        return info


class BinConv2d(nn.Conv2d):
    """Binarized 2d-convolutional layer.

    First layer should set quant_in_act=False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        quant_in_act: bool = False,
        clamp_weight: bool = False,
        scaling: bool = True,
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
        self.register_buffer('quant_in_act', torch.as_tensor(quant_in_act, dtype=torch.bool))
        self.register_buffer('clamp_weight', torch.as_tensor(clamp_weight, dtype=torch.bool))
        self.register_buffer('scaling', torch.as_tensor(scaling, dtype=torch.bool))
        self.register_buffer('scale', torch.as_tensor(0.0, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quant_in_act:
            input = BinActQuant.apply(input)

        if self.clamp_weight:
            with torch.no_grad():
                self.weight = self.weight.clamp(-1.0, 1.0)

        quant_weight = BinQuant.apply(self.weight)
        if self.scaling:
            if self.training:
                self.scale = self.get_scale()
            quant_weight = quant_weight * self.scale

        output = F.conv2d(
            input,
            quant_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def get_quant_weight(self) -> torch.Tensor:
        if self.scaling:
            if self.scale == 0.0:
                scale = self.weight.abs().mean()
            else:
                scale = self.scale
            return BinQuant.apply(self.weight) * scale
        else:
            return BinQuant.apply(self.weight)

    def get_scale(self) -> torch.Tensor:
        return self.weight.abs().mean()

    def __repr__(self) -> str:
        info = super().__repr__()[:-1] + f', quant_in_act={self.quant_in_act}' + f', clamp_weight={self.clamp_weight})'
        return info


if __name__ == "__main__":
    conv1 = BinConv2d(1, 3, 3)
    linear1 = BinLinear(10, 5)
