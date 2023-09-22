import glob
import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision import transforms

__all__ = ['WashingtonRGBDDataset']

RGBD_CLASSES = {
    'apple': 0,
    'ball': 1,
    'banana': 2,
    'bell_pepper': 3,
    'binder': 4,
    'bowl': 5,
    'calculator': 6,
    'camera': 7,
    'cap': 8,
    'cell_phone': 9,
    'cereal_box': 10,
    'coffee_mug': 11,
    'comb': 12,
    'dry_battery': 13,
    'flashlight': 14,
    'food_bag': 15,
    'food_box': 16,
    'food_can': 17,
    'food_cup': 18,
    'food_jar': 19,
    'garlic': 20,
    'glue_stick': 21,
    'greens': 22,
    'hand_towel': 23,
    'instant_noodles': 24,
    'keyboard': 25,
    'kleenex': 26,
    'lemon': 27,
    'lightbulb': 28,
    'lime': 29,
    'marker': 30,
    'mushroom': 31,
    'notebook': 32,
    'onion': 33,
    'orange': 34,
    'peach': 35,
    'pear': 36,
    'pitcher': 37,
    'plate': 38,
    'pliers': 39,
    'potato': 40,
    'rubber_eraser': 41,
    'scissors': 42,
    'shampoo': 43,
    'soda_can': 44,
    'sponge': 45,
    'stapler': 46,
    'tomato': 47,
    'toothbrush': 48,
    'toothpaste': 49,
    'water_bottle': 50,
}


def get_train_test_images_labels(
    dataset_dir: str, test_size: float, random_state: int
) -> Tuple[List[str], List[str], List[str], List[str], List[int], List[int]]:
    dataset_dir = os.path.expanduser(dataset_dir)
    folders = glob.glob(os.path.join(dataset_dir, '*'))
    rgbs, ds, labels = [], [], []

    for f in folders:
        basename = os.path.basename(f)
        folder = os.path.dirname(f)
        rgb = glob.glob(os.path.join(folder, basename, '*/*_crop.png'))
        d = glob.glob(os.path.join(folder, basename, '*/*_depthcrop.png'))

        label = RGBD_CLASSES[basename]
        label = [label for _ in range(len(d))]
        rgb.sort()
        d.sort()
        rgbs += rgb
        ds += d
        labels += label

    assert len(rgbs) == len(ds) or len(rgbs) != 0 or len(ds) != 0
    train_rgb, test_rgb, train_d, test_d, train_label, test_label = train_test_split(
        rgbs, ds, labels, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_rgb, test_rgb, train_d, test_d, train_label, test_label


class WashingtonRGBDDataset(Dataset):
    def __init__(
        self,
        rgb_imgs: List[str],
        d_imgs: List[str],
        labels: List[int],
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert len(rgb_imgs) == len(d_imgs)
        self.rgb_imgs = rgb_imgs
        self.d_imgs = d_imgs
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        rgb_img = cv2.imread(self.rgb_imgs[idx], cv2.IMREAD_COLOR)
        assert rgb_img is not None, f'Not found RGB image from: {rgb}'
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        d_img = cv2.imread(self.d_imgs[idx], cv2.IMREAD_ANYDEPTH)
        d_img = np.expand_dims(d_img, axis=-1)
        assert d_img is not None, f'Not found Depth image from: {d}'

        image = np.concatenate((rgb_img, d_img), axis=-1)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        label = self.labels[idx]
        return image, label

    def __len__(self) -> int:
        return len(self.labels)


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    (
        train_rgb,
        test_rgb,
        train_d,
        test_d,
        train_label,
        test_label,
    ) = get_train_test_images_labels('~/datasets/rgbd-dataset', test_size=0.3, random_state=0)

    transforms = A.Compose(
        [
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    dataset = WashingtonRGBDDataset(train_rgb, train_d, train_label, transforms)

    img, label = next(iter(dataset))
    print(img.shape, label)
