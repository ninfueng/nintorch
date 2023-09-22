import glob
import os
import tarfile
import zipfile
from typing import Callable, List, Optional, Tuple

import requests
from PIL import Image
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset

__all__ = ['CINIC10']


class CINIC10(Dataset):
    """CINIC10 dataset with a burst mode support.

    * ImageFolder: 29 seconds
    * CINIC10    : 20 seconds
    * Burst mode : 8  seconds

    Example:
    >>> dataset = CINIC10('~/datasets/cinic10', mode='train')
    """

    DATASET_URL = 'https://datashare.ed.ac.uk/download/DS_10283_3192.zip'
    CLASSES = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9,
    }
    NUM_CLASSES = len(CLASSES)

    def __init__(
        self,
        root: str,
        split: str,
        transforms: Optional[Callable[..., nn.Module]] = None,
        target_transforms: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert split in [
            'train',
            'test',
            'valid',
        ], f'`split` should be [`train`, `test`, `valid`], Your {split}.'
        self.split = split
        self.root = os.path.expanduser(root)

        if not os.path.exists(self.root):
            os.makedirs(self.root)
            self.download_dataset()

        data_dirs, self.labels = self.get_data_label_dirs()
        self.imgs = [Image.open(d).convert('RGB') for d in data_dirs]
        self.transforms = transforms
        self.target_transforms = target_transforms

    def download_dataset(self) -> None:
        """Download datasets + extract a zip file + extract a gzip +
        remove zip and gzip files.
        """

        zipname = os.path.basename(self.DATASET_URL)
        zipname = os.path.join(self.root, zipname)

        print(f'Downloading from: {self.DATASET_URL}. This may take a while.')
        response = requests.get(self.DATASET_URL)

        with open(zipname, 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile(zipname, 'r') as z:
            z.extractall(self.root)

        tarname = os.path.join(self.root, 'CINIC-10.tar.gz')
        with tarfile.open(tarname, 'r') as t:
            t.extractall(self.root)

        os.remove(zipname)
        os.remove(tarname)

    def get_data_label_dirs(self) -> Tuple[List[str], List[int]]:
        """Find a list image directories and a list of their labels."""
        data_dir = os.path.join(self.root, self.split)

        data_dirs, labels = [], []
        for k, v in self.CLASSES.items():
            tmp_dir = glob.glob(os.path.join(data_dir, k, '*.png'))
            data_dirs += tmp_dir
            tmp_label = [v for _ in tmp_dir]
            labels += tmp_label

        assert len(data_dirs) == len(labels) == 90_000
        return data_dirs, labels

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img, label = self.imgs[idx], self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label

    def __len__(self) -> int:
        return len(self.labels)


if __name__ == '__main__':
    cinic10 = CINIC10('./datasets/cinic10', 'train')
