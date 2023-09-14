import torch.nn as nn
from torchvision.models import resnet18

from nintorch.utils import convert_layer


class AnotherLinear(nn.Linear):
    """Just for replace `nn.Linear` to `AnotherLinear` for test with unittest."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestConvertLayer:
    def test_batchnorm(self) -> None:
        model = resnet18()
        model = convert_layer(model, nn.BatchNorm2d, nn.SyncBatchNorm)
        assert isinstance(model.bn1, nn.SyncBatchNorm)

    def test_linear(self) -> None:
        model = resnet18()
        model = convert_layer(model, nn.Linear, AnotherLinear)
        assert isinstance(model.fc, AnotherLinear)

    def test_conv(self) -> None:
        model = resnet18()
        model = convert_layer(model, nn.Conv2d, nn.ConvTranspose2d)
        assert isinstance(model.conv1, nn.ConvTranspose2d)
