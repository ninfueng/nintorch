import albumentations as A
import numpy as np
import torchvision.transforms as T
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

__all__ = ['Identity', 'get_basic_transforms', 'get_basic_albu_transforms']


class Identity(ImageOnlyTransform):
    def apply(self, img: np.ndarray, **_) -> np.ndarray:
        return img


def get_basic_transforms(
    resize: int = 256,
    crop: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=mean, std=std)
    train_transforms = T.Compose(
        [
            T.RandomResizedCrop(crop),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(crop),
            T.ToTensor(),
            normalize,
        ]
    )
    return (train_transforms, val_transforms)


def get_basic_albu_transforms(
    resize: int = 256,
    crop: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[A.Compose, A.Compose]:
    assert resize > crop, f'`resize` should more than `crop` {crop} > {resize}.'
    normalize = A.Normalize(mean=mean, std=std)
    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(crop, crop),
            A.HorizontalFlip(p=0.5),
            normalize,
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(resize, resize),
            A.CenterCrop(crop, crop),
            normalize,
            ToTensorV2(),
        ]
    )
    return (train_transforms, val_transforms)
