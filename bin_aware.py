from torch import Tensor, nn


def get_bin_aware_loss(model: nn.Module, max_range: float = 2.0, alpha: float = 1e-4) -> Tensor:
    abs_losses = 0.0
    for name, param in model.named_parameters():
        if name.endswith('weight'):
            if param.ndim < 2:
                continue
            abs_param = param.abs()
            abs_loss = max_range - abs_param[abs_param < max_range]
            abs_losses += abs_loss.mean()
    abs_losses *= alpha
    return abs_losses


class BinAwareCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, model: nn.Module, max_range: float = 2.0, alpha: float = 1e-4):
        super().__init__()
        self.model = model
        self.max_range = max_range
        self.alpha = alpha

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)
        bin_aware_loss = get_bin_aware_loss(self.model, self.max_range, self.alpha)
        return loss + bin_aware_loss


if __name__ == '__main__':
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    loss = get_bin_aware_loss(model, max_range=2.0)
    print(loss)
