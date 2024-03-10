from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18

__all__ = ['get_name_from_layer_types', 'input_act_hook']


def get_name_from_layer_types(
    model: nn.Module, types: Tuple[nn.Module, ...]
) -> List[str]:
    """Get all name of layers that has a type in `types`.

    Example:
    >>> model = resnet18(pretrained=True)
    >>> get_name_from_layer_types(model, (nn.Conv2d, ))
    """
    name_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, tuple(types)):
            name_layers.append(name)
    return name_layers


def input_act_hook(
    input_dict: Dict[str, List[Tensor]],
    act_dict: Dict[str, List[Tensor]],
    name: str,
    only: Optional[str] = None,
) -> Callable[[nn.Module, Tensor, Tensor], None]:
    """Add hooks to collect input and activation input `input_dict` and `act_dict`.
    https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/

    Example:
    >>> model = resnet18(pretrained=True)
    >>> model.layer1.register_forward_hook(act_hook('layer1'))
    >>> model(torch.randn(1, 3, 224, 224))
    """

    def hook(_: nn.Module, input: Tensor, act: Tensor) -> None:
        assert only is None or only in ('input', 'act')
        if only is None or only == 'input':
            input = input[0] if isinstance(input, tuple) else input
            if input_dict.get(name) is None:
                input_dict[name] = [input.detach()]
            else:
                input_dict[name].append(input.detach())
        if only is None or only == 'act':
            if act_dict.get(name) is None:
                act_dict[name] = [act.detach()]
            else:
                act_dict[name].append(act.detach())

    return hook


if __name__ == '__main__':
    from nincore import multi_getattr

    input_dict, act_dict = {}, {}
    model = resnet18(pretrained=True)
    model.eval()

    target_layers = get_name_from_layer_types(model, (nn.Conv2d,))
    for target_layer in target_layers:
        target_layer_hook = f'{target_layer}.register_forward_hook'
        reg_hook_method = multi_getattr(model, target_layer_hook)
        reg_hook_method(input_act_hook(input_dict, act_dict, target_layer))

    input = torch.randn(1, 3, 224, 224)
    model(input)
    input1 = torch.randn(1, 3, 224, 224)
    model(input1)

    for name, input in input_dict.items():
        input = input[0] if isinstance(input, list) else input
        input = input.detach().cpu().numpy()
        print(f'{name=} {input.shape=}')

    for name, act in act_dict.items():
        act = act[0] if isinstance(act, list) else act
        act = act.detach().cpu().numpy()
        print(f'{name=} {act.shape=}')
