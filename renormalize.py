"""renormalize by multiplication with large number to avoid bit-flipping
divided output with the same number to keep the same scale

this cannot be used with bias and batch normalization?
however, merge batch normalization is possible.
sigmoid is not work without associative property.
relu is work.

Preferred range: (-2, 1] or [1, 2), however this may got Inf problems.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import vgg11_bn


def get_minmax(model: nn.Module, thres: float = 2.0) -> Dict[str, Tuple[float, float]]:
    minmax_dict = {}
    for name, param in model.named_parameters():
        if param.dim() <= 1:
            continue
        maxmin_radio = abs(param.max().item() / param.min().item())
        if maxmin_radio <= thres:
            minmax_dict[name] = (param.min().item(), param.max().item())
    return minmax_dict


def get_scale_dict(model: nn.Module, thres: float = 2.0) -> Dict[str, float]:
    scale_dict = {}
    prev_scale = 1.0
    for name, param in model.named_parameters():
        if param.dim() <= 1:
            continue
        maxmin_radio = abs(param.max().item() / param.min().item())

        if maxmin_radio <= thres:
            scale = max(1.0 / param.max().abs().item(), 1.0 / param.min().abs().item())
            scale_dict[name] = scale
            prev_scale *= scale

            bias_name = name.replace('.weight', '.bias')
            scale_dict[bias_name] = prev_scale
            # print(f'`{name}`, maxmin_radio={maxmin_radio}')
        else:
            # print(f'skip `{name}`, maxmin_radio={maxmin_radio} > {thres}')

            bias_name = name.replace('.weight', '.bias')
            scale_dict[bias_name] = prev_scale
    return scale_dict


@torch.no_grad()
def apply_scale_dict(model: nn.Module, scale_dict: Dict[str, float]) -> None:
    for name, param in model.named_parameters():
        if name in scale_dict:
            param.mul_(scale_dict[name])


def rescale_output(model: nn.Module, output: torch.Tensor, scale_dict: Dict[str, float]) -> torch.Tensor:
    all_scale = 1.0
    for name, _ in model.named_parameters():
        if name in scale_dict:
            if name.endswith('.weight'):
                scale = scale_dict[name]
                all_scale *= scale
    return output / all_scale


if __name__ == '__main__':
    from pprint import pprint

    from nineff.utils.fuse import fuse_conv_bn

    # model = resnet18(pretrained=True)
    # model = vgg11(pretrained=True)
    model = vgg11_bn(pretrained=True)
    # model = mobilenet_v2(pretrained=True)
    # model = convnext_base(pretrained=True)
    model = fuse_conv_bn(model)

    model = model.eval()
    input = torch.rand(1, 3, 224, 224)
    ref = model(input)

    scale_dict = get_scale_dict(model)
    pprint(scale_dict)
    apply_scale_dict(model, scale_dict)
    model = model.eval()
    output = model(input)
    output = rescale_output(model, output, scale_dict)
    torch.testing.assert_close(ref, output)

    pprint(get_minmax(model))
