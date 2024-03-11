import torch
from torch import Tensor

__all__ = ['find_cross_weight']


def find_cross_weight(labels: Tensor) -> Tensor:
    """Find the cross-entropy weight for each class.
    More frequent class will cause the lower weight.

    Args:
        labels: class labels

    Returns:
        nor_cross_weight: cross-entropy weight in range of [0, 1].

    Example:
    >>> x = torch.tensor([0, 0, 1, 2, 3, 4])
    >>> find_cross_weight(x)
    tensor([0.5000, 1.0000, 1.0000, 1.0000, 1.0000])
    """
    n_classes = labels.unique().size(0)
    class_count = torch.bincount(labels, minlength=n_classes).float()
    cross_weight = class_count.sum() / class_count

    check_inf = torch.isinf(cross_weight)
    assert not check_inf.any(), (
        'Detect `inf` during normalization. '
        'Expected all class labels to be non-zero. '
        'Please check the input labels.'
    )
    nor_cross_weight = cross_weight / cross_weight.max()
    return nor_cross_weight
