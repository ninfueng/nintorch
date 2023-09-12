from typing import Sequence

import torch
from torch import nn
from torchprofile import profile_macs

__all__ = ["count_params", "count_macs"]


# MIT 6.5940 EfficientML.ai Fall 2023: Lab 0 PyTorch Tutorial
def count_params(model: nn.Module) -> int:
    """Count number of learn-able parameters in a model."""
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    return num_params


# While can using with `torchprofile.count_mac` directly, but just for a reminder.
def count_macs(
    model: nn.Module,
    input_size: Sequence[int],
    device: torch.device = torch.device("cuda"),
) -> int:
    model = model.to(device)
    input = torch.empty(input_size, device=device)
    macs = profile_macs(model, input)
    return macs


if __name__ == "__main__":
    from torchvision.models import resnet18

    model = resnet18()
    mac = count_macs(model, (1, 3, 224, 224))
    print(mac)
