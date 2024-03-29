import os
from typing import Any, Callable

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

__all__ = ['PreloadImageFolder', 'PreloadListDataset', 'cv_loader', 'pil_loader']


pil_loader = default_loader


def cv_loader(img_dir: str) -> np.ndarray:
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class PreloadListDataset(Dataset):
    """A general dataset to load images from a list of image directories and labels.

    Arguments:
        img_labels: a list of tuples, each tuple contains an image directory and its label.
        transform: to transforms all data after loading.
        target_transform: to transforms all labels after loading.

    Example:
    >>> img_dirs = ['~/datasets/imagenet/train/n01440764/n01440764_10026.JPEG']
    >>> labels = [0]
    >>> dataset = ListDataset(img_dirs, labels, transform=transforms.ToTensor())
    >>> img, label = next(iter(dataset))
    """

    def __init__(
        self,
        img_dirs: list[str],
        labels: list[Any],
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = default_loader,
        albu: bool = False,
        preload: bool = False,
    ) -> None:
        assert len(img_dirs) == len(
            labels
        ), f'Size of `img_dirs` != `labels`, {len(img_dirs)=} != {len(labels)=}'
        self.img_dirs = [os.path.expanduser(img_dir) for img_dir in img_dirs]
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.albu = albu
        self.imgs = None
        self.preload = preload
        if preload:
            self.imgs = [self.loader(img_dir) for img_dir in img_dirs]

    def __getitem__(self, idx: int) -> Any:
        if self.preload:
            img = self.imgs[idx]
        else:
            img_dir = self.img_dirs[idx]
            img = self.loader(img_dir)
        label = self.labels[idx]

        if self.transform is not None:
            if self.albu:
                transformed = self.transform(image=img)
                img = transformed['image']
            else:
                img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self) -> int:
        return len(self.labels)


class PreloadImageFolder(ImageFolder):
    """Load all images into a list, however this may consume a large amount of RAM memory.

    Arguments:
        transforms_first: to transforms all data after loading.

    Example:
    >>> imagefolder = PreloadImageFolder('~/datasets/imagenet/val/')
    >>> img, label = next(iter(imagefolder))
    >>> print(img, label)
    """

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        transform_first: bool = False,
        albu: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        imgs = self.samples
        pbar = tqdm(imgs)
        pbar.set_description('PreloadImageFolder')

        def _wrap_albu_transforms(img: np.ndarray) -> np.ndarray:
            transformed = transform(image=img)
            img = transformed['image']
            return img

        img_labels = []
        for img_dir, label in pbar:
            img = self.loader(img_dir)

            if transform_first and transform is not None:
                if albu:
                    img = _wrap_albu_transforms(img)
                else:
                    img = transform(img)

            if transform_first and target_transform is not None:
                label = target_transform(label)
            img_labels.append((img, label))

        # already used a `loader` not need to use again.
        self.loader = lambda x: x
        self.samples = img_labels

        if transform_first:
            self.transform = lambda x: x
            self.target_transform = lambda x: x
        else:
            if transform is not None:
                self.transforms = _wrap_albu_transforms


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms

    from nintorch import torch_np

    # imagefolder = PreloadImageFolder('~/datasets/imagenet/val', transform_first=True)
    # img, label = next(iter(imagefolder))
    # print(img, label)

    img_dirs = ['~/datasets/imagenet/train/n01440764/n01440764_10026.JPEG']
    labels = [0]
    dataset = PreloadListDataset(img_dirs, labels, transform=transforms.ToTensor())
    img, label = next(iter(dataset))
    img = torch_np(img)

    plt.imshow(img)
    plt.show()
