import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        with_bn: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        with_bias = not with_bn
        try:
            activation_fn(inplace=True)
            with_inplace = True
        except TypeError:
            with_inplace = False

        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, bias=with_bias),
            nn.BatchNorm2d(6) if with_bn else nn.Identity(),
            activation_fn(inplace=True) if with_inplace else activation_fn(),
            nn.MaxPool2d(2, 2),
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(6, 16, 5, bias=with_bias),
            nn.BatchNorm2d(16) if with_bn else nn.Identity(),
            activation_fn(inplace=True) if with_inplace else activation_fn(),
            nn.MaxPool2d(2, 2),
        )

        self.l2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120, bias=with_bias),
            nn.BatchNorm1d(120) if with_bn else nn.Identity(),
            activation_fn(inplace=True) if with_inplace else activation_fn(),
        )

        self.l3 = nn.Sequential(
            nn.Linear(120, 84, bias=with_bias),
            nn.BatchNorm1d(84) if with_bn else nn.Identity(),
            activation_fn(inplace=True) if with_inplace else activation_fn(),
        )

        self.linear = nn.Linear(84, num_classes, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.l0(input)
        output = self.l1(output)
        output = self.l2(output)
        output = self.l3(output)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    model = LeNet5()
    print(model)
    model = model.eval()
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    print(output.shape)
