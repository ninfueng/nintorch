"""Generate randomly selected subset of indexs and save in `.pt` file.

Note:
* ImageNet train size: 1,281,167
* CIFAR10 train size: 50,000
* CIFAR100 train size: 50,000
* CINIC10 train size: 90,000
"""
import argparse
import os

import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate subset index for training.')
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--train-size', type=int, default=1_281_167)
    parser.add_argument('--subset-idx-save-dir', type=str, default='subset_idx.pt')
    args = parser.parse_args()

    subset_idx = np.random.choice(args.train_size, size=int(args.train_size * args.fraction))
    subset_idx = torch.as_tensor(subset_idx, dtype=torch.int32)
    print(f'Set subset index with following shape: `{subset_idx.shape}`')

    subset_idx_save_dir = os.path.expanduser(args.subset_idx_save_dir)
    dirname = os.path.dirname(subset_idx_save_dir)
    try:
        os.makedirs(dirname, exist_ok=True)
    except FileNotFoundError:
        print('Skip `os.makedirs` to `os.path.dirname(subset_idx_save_dir)`')
    torch.save(subset_idx, subset_idx_save_dir)
    print(f'Save subset index to `{subset_idx_save_dir}`.')
