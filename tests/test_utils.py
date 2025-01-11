import torch
from torchvision.models import resnet50

from nintorch.utils import get_gpu_usage

if torch.cuda.is_available():

    def test_get_gpu_usage() -> None:
        model = resnet50()
        device = torch.device('cuda:0')
        model = model.to(device)
        mem = get_gpu_usage(device)
        assert mem > 0
