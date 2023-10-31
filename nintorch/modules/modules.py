from typing import Callable, Optional, Union

import torch.nn as nn
from torch import Tensor

__all__ = [
    'Lambda',
    'ConvNormAct',
    'LinearNormAct',
]


class Lambda(nn.Module):
    """Lambda module to wrap a function with.

    Arguments:
        fn: a function to apply during forward propagation.
    """

    def __init__(self, fn: Callable[..., Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        x = self.fn(x)
        return x

    def __repr__(self) -> str:
        return f'{super().__repr__()[:-1]}fn={self.fn})'


class ConvNormAct(nn.Sequential):
    """Convolutional + Normalization + Activation Layers.

    Allow to access quantization after normalization and activation functions.
    All `conv`, `norm`, and `act` can be disabled via using `None` as input.
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
        bias: bool = False,
        padding_mode: str = 'zeros',
        conv: Callable[..., nn.Module] = nn.Conv2d,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        act: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        layers = nn.ModuleList()
        conv = conv(
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
        layers.append(conv)
        if norm is not None:
            layers.append(norm(out_channels))
        if act is not None:
            try:
                act = act(inplace=True)
            except TypeError:
                act = act()
            layers.append(act)
        super().__init__(*layers)


class LinearNormAct(nn.Sequential):
    """Linear + Normalization + Activation Layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        linear: Callable[..., nn.Module] = nn.Linear,
        norm: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
        act: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        layers = nn.ModuleList()
        linear = linear(in_features, out_features, bias)
        layers.append(linear)
        if norm is not None:
            layers.append(norm(out_features))
        if act is not None:
            try:
                act = act(inplace=True)
            except TypeError:
                act = act()
            layers.append(act)
        super().__init__(*layers)


if __name__ == '__main__':
    import torch

    class TestModule:
        def test_conv_norm_act(self) -> None:
            x = torch.randn(1, 3, 32, 32)
            conv = ConvNormAct(3, 6, 3)
            conv = conv.eval()
            output = conv(x)
            assert output.shape == (1, 6, 30, 30)

        def test_linear_norm_act(self) -> None:
            x = torch.randn(1, 10)
            linear = LinearNormAct(10, 1)
            linear = linear.eval()
            output = linear(x)
            assert output.shape == (1, 1)
