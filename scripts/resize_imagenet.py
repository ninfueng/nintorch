import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import torchvision.transforms as T
from PIL import Image

jpeg_png_val_transforms = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
    ]
)

pt_val_transforms = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def resize_save(path: Path, dest: Path, mode: str, filename: Path) -> None:
    img = Image.open(filename)
    new_filename = dest / filename.relative_to(path)
    new_filename.parent.mkdir(exist_ok=True)

    # Original: 0.69764, 6.4 GB, 15.599 seconds.
    if mode == 'jpeg':
        img = jpeg_png_val_transforms(img)
        # Max quality of JPEG, but still causes loss from  to 0.69676.
        # 0.69676, 3.5 GB, 10.202 seconds.
        # https://jdhao.github.io/2019/07/20/pil_jpeg_image_quality/
        img.save(new_filename, quality=100, subsampling=0, format='JPEG')

    elif mode == 'pt':
        # 0.69884, 28 GB from 6.4 GB, 9.8915 second
        new_filename = new_filename.with_suffix('.PT')
        img = pt_val_transforms(img)
        with open(new_filename, 'wb') as f:
            pickle.dump(img, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif mode == 'png':
        # 0.69757, 4.2 GB from 6.4 GB, 13.605 second
        new_filename = new_filename.with_suffix('.PNG')
        img = jpeg_png_val_transforms(img)
        img.save(new_filename, format='PNG')

    else:
        raise NotImplementedError(f'Support only `jpeg`, `pt`, `png`. Your: {mode}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--imagenet-dir', type=str, default='~/datasets/imagenet/val')
    parser.add_argument('--resized-imagenet-dir', type=str, default='~/datasets/resized-imagenet')
    parser.add_argument('--mode', type=str, default='jpeg')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    imagenet_dir = Path(args.imagenet_dir).expanduser()
    resized_imagenet_dir = Path(args.resized_imagenet_dir).expanduser()
    resized_imagenet_dir.mkdir(exist_ok=True)
    files = imagenet_dir.glob('*/*.JPEG')

    t0 = time.perf_counter()
    print(f'Resizing, might take a while.')
    with ProcessPoolExecutor(args.workers) as e:
        e.map(partial(resize_save, imagenet_dir, resized_imagenet_dir, args.mode), files)
    diff = time.perf_counter() - t0
    print(f'Done, runtime: {diff} seconds')
