import glob
import logging
import os
import tarfile
import zipfile
from typing import Callable, List, Optional, Tuple

import requests
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__file__)

__all__ = ["CINIC10"]


class CINIC10(Dataset):
    """CINIC10 dataset with a burst mode support.
    ImageFolder 29 seconds, CINIC10 20 seconds, and Burst mode 8 seconds.

    Example:
    >>> dataset = CINIC10("~/datasets/cinic10", mode="train")
    """

    DATASET_URL = "https://datashare.ed.ac.uk/download/DS_10283_3192.zip"
    CLASSES = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    NUM_CLASSES = len(CLASSES)

    def __init__(
        self, root: str, mode: str, transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()
        assert isinstance(root, str)
        assert isinstance(mode, str)
        mode = mode.lower()
        assert mode in ["train", "test", "valid"]
        self.mode = mode

        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            self.download_dataset()

        data_dirs, self.labels = self.get_data_label_dirs()
        self.imgs = [Image.open(d).convert("RGB") for d in data_dirs]
        self.transforms = transforms

    def download_dataset(self) -> None:
        zip_file = os.path.basename(self.DATASET_URL)
        zip_file = os.path.join(self.root, zip_file)
        logger.info(f"Downloading from {self.DATASET_URL}. This may take a while.")

        response = requests.get(self.DATASET_URL)
        with open(zip_file, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(self.root)

        tarname = os.path.join(self.root, "CINIC-10.tar.gz")
        with tarfile.open(tarname, "r") as t:
            t.extractall(self.root)
        os.remove(zip_file)
        os.remove(tarname)

    def get_data_label_dirs(self) -> Tuple[List[str], List[int]]:
        data_dir = os.path.join(self.root, self.mode)
        data_dirs, labels = [], []

        for k, v in self.CLASSES.items():
            tmp_dir = glob.glob(os.path.join(data_dir, k, "*.png"))
            data_dirs += tmp_dir
            tmp_label = [v for _ in tmp_dir]
            labels += tmp_label

        assert len(data_dirs) == len(labels) == 90_000
        return data_dirs, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img, label = self.imgs[idx], self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
