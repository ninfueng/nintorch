import os

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

__all__ = [
    'save_safetensors',
    'load_safetensors',
]


def save_safetensors(
    data_dict: dict[str, Tensor],
    save_dir: str,
) -> None:
    """Example:

    >>> model = resnet18()
    >>> model_save_dir = 'resnet18.safetensors'
    >>> save_safetensors(model.state_dict(), model_save_dir)
    """
    save_dir = os.path.expanduser(save_dir)
    save_file(data_dict, save_dir)


def load_safetensors(load_dir: str) -> dict[str, Tensor]:
    """Example:

    >>> model = resnet18()
    >>> model_save_dir = 'resnet18.safetensors'
    >>> save_safetensors(model.state_dict(), model_save_dir)
    >>> model_state_dict = load_safetensors(model_save_dir)
    >>> model.load_state_dict(model_state_dict)
    """
    load_dir = os.path.expanduser(load_dir)
    data_dict = {}
    with safe_open(load_dir, framework='pt', device='cpu') as f:
        for k in f.keys():
            data_dict[k] = f.get_tensor(k)
    return data_dict
