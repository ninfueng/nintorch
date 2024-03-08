import torch
from torch import nn

from nintorch.modules import ConvNormAct, LinearNormAct


class TestModules:
    @torch.inference_mode()
    def test_conv_norm_act(self) -> None:
        x = torch.randn(1, 3, 32, 32)
        conv = ConvNormAct(3, 6, 3)
        conv = conv.eval()
        output = conv(x)
        assert output.shape == (1, 6, 30, 30)

    @torch.inference_mode()
    def test_linear_norm_act(self) -> None:
        x = torch.randn(1, 10)
        linear = LinearNormAct(10, 1)
        linear = linear.eval()
        output = linear(x)
        assert output.shape == (1, 1)

    def test_order_less_than_three(self) -> None:
        cn = ConvNormAct(3, 6, 3, order='cn')
        la = LinearNormAct(10, 1, order='la')
        assert len(cn) == 2
        assert len(la) == 2

    def test_order_three(self) -> None:
        cn = ConvNormAct(3, 6, 3, order='cna')
        la = LinearNormAct(10, 1, order='lna')
        assert len(cn) == 3
        assert len(la) == 3

    def test_anc(self) -> None:
        anc = ConvNormAct(3, 6, 3, order='anc')
        anl = LinearNormAct(10, 1, order='anl')
        assert len(anc) == 3
        assert len(anl) == 3
        assert isinstance(anc[0], nn.ReLU)
        assert isinstance(anl[0], nn.ReLU)
        assert isinstance(anc[1], nn.BatchNorm2d)
        assert isinstance(anl[1], nn.BatchNorm1d)
        assert isinstance(anc[2], nn.Conv2d)
        assert isinstance(anl[2], nn.Linear)
