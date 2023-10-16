from typing import List, Optional

import numpy as np
import numpy2ap as npa
import torch
from nincore import multi_setattr
from torch import nn


def corrupt_bin(bins: np.ndarray, p: float = 1e-2, excepts: Optional[List[int]] = None) -> np.ndarray:
    """Randomly bit-flipping binary array with `p` probability to toggle bit.

    In the same 32-bit floating point, there may 0-4 bit corrupted in models.

    Arguments:
        bins: a binary representation array to corrupt.
        p: a probability to corrupt bit or produce `1`.

    Returns:
        corrupted_bins: a corrupted binary representation array.
    """
    assert 0.0 <= p <= 1.0, f'p: {p} should be in (0, 1].'
    noise_mask = np.random.choice(2, bins.shape, p=[1.0 - p, p])
    if excepts is not None:
        noise_mask[:, excepts] = 0
    noise_mask = noise_mask.astype(bool)
    corrupted_bins = np.logical_xor(bins, noise_mask)
    return corrupted_bins


def sim_write_error_protection(
    model: nn.Module,
    wer: float = 1e-2,
    excepts: Optional[List[int]] = None,
) -> nn.Module:
    """Simulate write-error rate (WER) and repetition code protection.
    Tests this protection with the random-bit flipping with `wer` rates.

    Arguments:
        model: a model with parameters to protect and to bit-flipping.
        repeats: list of number of duplicate bits for the protection with repetition code.
        wer: write error rate or a fraction of bit-flipping to parameters.

    Return:
        model: model with bit-flipping.
    """
    for name, param in model.named_parameters():
        if name == 'classifier.0.weight':
            continue
        param = param.detach().cpu().numpy()
        origin_shape = param.shape
        bin_ = npa.float_bin(param)
        bin_ = bin_.reshape(-1, 32)

        protect = corrupt_bin(bin_, wer, excepts)
        noisy_param = npa.bin_float(protect)
        noisy_param = torch.as_tensor(noisy_param)
        noisy_param = noisy_param.reshape(origin_shape)
        noisy_param = nn.Parameter(noisy_param)
        multi_setattr(model, name, noisy_param)
    return model


if __name__ == '__main__':
    input = np.ones(100, dtype=np.float32)
    input = npa.float_bin(input)
    input = corrupt_bin(input, p=0.1, excepts=[1])
    output = npa.bin_float(input)
    print(output)
