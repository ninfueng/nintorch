from typing import Any, Callable, Optional

import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

__all__ = ['PreloadImageFolder', 'cv_loader', 'pil_loader']

pil_loader = default_loader


def cv_loader(img_dir: str) -> np.ndarray:
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# TODO: add supports to albumentations.
class PreloadImageFolder(ImageFolder):
    """Load all images into a list, but may consume a high RAM memory.

    Example:
    >>> imagefolder = PreloadImageFolder('~/datasets/imagenet/val/')
    >>> img, label = next(iter(imagefolder))
    >>> print(img, label)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        transform_first: bool = False,
    ) -> None:
        # Using
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        imgs = self.samples
        pbar = tqdm(imgs)
        pbar.set_description('PreloadImageFolder')

        img_labels = []
        for img_dir, label in pbar:
            img = self.loader(img_dir)
            if transform_first and transform is not None:
                img = transform(img)
            if transform_first and target_transform is not None:
                label = target_transform(label)
            img_labels.append((img, label))

        # Already used a `loader` not need to use again.
        self.loader = lambda x: x
        self.samples = img_labels
        if transform_first:
            self.transform = lambda x: x
            self.target_transform = lambda x: x


if __name__ == '__main__':
    imagefolder = PreloadImageFolder('~/datasets/imagenet/val', transform_first=True)
    img, label = next(iter(imagefolder))
    print(img, label)
