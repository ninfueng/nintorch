import torch.nn as nn
from torch import Tensor

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        'M',
    ],
    'VGG19': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        512,
        'M',
    ],
}


class VGG7(nn.Module):
    """VGG7 model described in Ternary Weight Networks [1, 2].

    Full-precision should have 92.88% accuracy on CIFAR-10?
    Ternary should have 92.56% accuracy on CIFAR-10?

    [1]: https://arxiv.org/abs/1605.04711
    [2]: https://raw.githubusercontent.com/Thinklab-SJTU/twns/main/cls/litenet.py
    """

    def __init__(
        self, num_classes: int = 10, bias: bool = False, last_bn: bool = False
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 1_024, bias=bias),
            nn.BatchNorm1d(1_024),
            nn.ReLU(inplace=True),
            nn.Linear(1_024, num_classes, bias=True),
        )
        # https://arxiv.org/pdf/2007.14234.pdf
        # Using BN after a last layer.
        self.last_bn = nn.BatchNorm1d(num_classes) if last_bn else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.linear(x)
        if self.last_bn is not None:
            x = self.last_bn(x)
        return x


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
