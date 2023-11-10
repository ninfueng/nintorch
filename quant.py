from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

__all__ = ['BinQuant', 'BinActivationQuant', 'BinConv2d', 'BinLinear']


def get_thres(weight: Tensor, delta: float = 0.75) -> Union[float, Tensor]:
    """Find a ternary threshold as stated in `Ternary Weight Networks` paper.

    2023 version of paper changes to 0.75 L1 norm of floating point weights.

    Arguments:
        input: an input tensor.
        delta: a constant in range of (0.0, 1.0].

    Returns:
        thres: a threshold.
    """
    assert 0.0 < delta <= 1.0, f'`delta` should be in range (0.0, 1.0], Your {delta}.'
    avg = weight.abs().sum() / weight.numel()
    thres = avg * delta
    return thres


class TerQuant(Function):
    """Ternary quantization with a threshold."""

    @staticmethod
    def forward(_, input: Tensor, thres: Tensor, scale: Optional[Tensor] = None) -> Tensor:
        neg_mask = (input < -thres).float()
        pos_mask = (input > thres).float()
        if scale is not None:
            output = (pos_mask - neg_mask) * scale
        else:
            output = pos_mask - neg_mask
        return output

    @staticmethod
    def backward(_, grad_out: Tensor) -> Tuple[Tensor, None, None]:
        grad_input = grad_out.clone()
        return (grad_input, None, None)


class BinQuant(torch.autograd.Function):
    """Designed to use with weights."""

    @staticmethod
    def forward(_: Any, input: torch.Tensor) -> torch.Tensor:
        return input.sign()

    @staticmethod
    def backward(_: Any, grad_out: torch.Tensor) -> torch.Tensor:
        """Replace backward process with do nothings instead.
        Gradients come from output -> input.
        """
        grad_input = grad_out.clone()
        return grad_input


class BinActivationQuant(torch.autograd.Function):
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

    First layer should set quant_in_activation=False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_in_activation: bool = False,
        clamp_weight: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.quant_weight = None
        self.quant_in_activation = quant_in_activation
        self.clamp_weight = clamp_weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quant_in_activation:
            input = BinActivationQuant.apply(input)

        if self.clamp_weight:
            with torch.no_grad():
                self.weight = self.weight.clamp(-1, 1)

        self.quant_weight = BinQuant.apply(self.weight)
        output = F.linear(input, self.quant_weight, self.bias)
        return output

    def get_quant_weight(self) -> torch.Tensor:
        if self.quant_weight is None:
            return BinQuant.apply(self.weight)
        return self.quant_weight

    def __repr__(self) -> str:
        info = (
            super().__repr__()[:-1]
            + f' quant_in_activation={self.quant_in_activation}), '
            + f'clamp_weight={self.clamp_weight})'
        )
        return info


class BinConv2d(nn.Conv2d):
    """Binarized 2d-convolutional layer.

    First layer should set quant_in_activation=False.
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
        padding_mode: str = 'zeros',
        quant_in_activation: bool = True,
        clamp_weight: bool = False,
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
        self.quant_in_activation = quant_in_activation
        self.clamp_weight = clamp_weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quant_in_activation:
            input = BinActivationQuant.apply(input)

        if self.clamp_weight:
            with torch.no_grad():
                self.weight = self.weight.clamp(-1, 1)

        self.quant_weight = BinQuant.apply(self.weight)

        output = F.conv2d(
            input,
            self.quant_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def get_quant_weight(self) -> torch.Tensor:
        if self.quant_weight is None:
            return BinQuant.apply(self.weight)
        return self.quant_weight

    def __repr__(self) -> str:
        info = (
            super().__repr__()[:-1]
            + f' quant_in_activation={self.quant_in_activation}), '
            + f'clamp_weight={self.clamp_weight})'
        )
        return info


if __name__ == '__main__':
    conv1 = BinConv2d(1, 3, 3)
    linear1 = BinLinear(10, 5)
