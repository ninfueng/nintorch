import torch

from nintorch.modules import ConvNormAct, LinearNormAct


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
