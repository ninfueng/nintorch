"""Spatial Pyramid Pooling allows to accept multi-resolution images.

However, since `PyTorch` model accepts only a fixed-size batch Tensor.
In paper, train an epoch of 224x224 images and switch to an epoch of 180x180 images.
"""
import logging
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__file__)

try:
    from torch.jit import script

    logger.warn(
        "Cannot import `torch.jit.script`. Wrapped with identity wrapper instead."
    )

except ImportError:
    from nincore.wrap import wrap_identity

    # Money padding if cannot use torch.jit.script then wrapping with identity wrapper.
    script = wrap_identity


@script
def spatial_pyramid_pool2d(
    input: Tensor, bins: Union[int, List[int]], mode: str = "max"
) -> Tensor:
    """Spatial Pyramid Pooling: https://arxiv.org/pdf/1406.4729.pdf

    Args:
        input (Tensor): an input tensor expected from the convolutional layer.
        bins (List[int]): a list of integer of preferred size of outputs.
        mode (str): how to reduce the spatial space.

    Returns:
        outputs (Tensor): a flatten tensor with size (batch, bins[0] * bins[0] + bins[1]
        * bins[1] + ...)
    """
    mode = mode.lower()
    b, _, h, w = input.shape
    bins = [bins] if isinstance(bins, int) else bins
    outputs = []

    for b in bins:
        h_kernel, w_kernel = math.ceil(h / b), math.ceil(w / b)
        h_stride, w_stride = math.floor(h / b), math.floor(w / b)
        if mode == "max":
            output = F.max_pool2d(
                input, kernel_size=(h_kernel, w_kernel), stride=(h_stride, w_stride)
            )
        elif mode == "avg" or mode == "average" or mode == "mean":
            output = F.avg_pool2d(
                input, kernel_size=(h_kernel, w_kernel), stride=(h_stride, w_stride)
            )
        else:
            raise NotImplementedError(
                "`mode` only accepts `max`, `avg`, `average` and `mean` only. "
                f"Your `mode`: {mode}"
            )
        output = output.flatten(start_dim=1)
        outputs.append(output)

    outputs = torch.cat(outputs, dim=-1)
    return outputs


class SpatialPyramidPool2d(nn.Module):
    def __init__(self, bins: Union[int, List[int]], mode: str = "max") -> None:
        super().__init__()
        self.bins = bins
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return spatial_pyramid_pool2d(input, bins=self.bins, mode=self.mode)


def spp_collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[List[Tensor], Tensor]:
    batch_images, batch_labels = [], []
    for images, labels in batch:
        batch_images.append(images)
        batch_labels.append(labels)
    batch_labels = torch.as_tensor(batch_labels)
    return batch_images, batch_labels


if __name__ == "__main__":
    input = torch.zeros(1, 512, 13, 13)
    output = spatial_pyramid_pool2d(input, [1, 2, 3], "max")
    print(output.shape)

    spp = SpatialPyramidPool2d([1, 2, 3], "max")
    output2 = spp(input)
    torch.testing.assert_close(output, output2)
