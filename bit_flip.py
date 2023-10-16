from typing import Dict

import numpy as np
import numpy2ap as npa
import torch
import torch.nn as nn
from nincore import AttrDict
from nineff.low.ternary import TerConv2d, TerLinear

from bin_quant import BinConv2d, BinLinear
from nintorch.models import construct_model_cifar
from nintorch.utils import convert_layer

to_numpy = lambda x: x.detach().cpu().numpy()


def sign_to_bin(input: np.ndarray) -> np.ndarray:
    pos = np.sign(input) == 1
    return pos


def ter_to_bin(input: np.ndarray, codeword: Dict[int, np.ndarray]) -> np.ndarray:
    shape = list(input.shape) + [2]

    bin = np.zeros(shape, dtype=bool)
    for c in codeword:
        bin[input == c] = codeword[c]
    return bin


def bin_to_ter(input: np.ndarray, codeword: Dict[int, np.ndarray]) -> np.ndarray:
    shape = input.shape[:-1]
    ter = np.zeros(shape, dtype=int)
    for c in codeword:
        ter[np.all(input == codeword[c], axis=-1)] = c
    return ter


def bin_to_sign(input: np.ndarray) -> np.ndarray:
    sign = np.ones(input.shape, dtype=np.int32)
    sign[input == False] = -1
    return sign


def inject_bit_flip(input: np.ndarray, p: float) -> np.ndarray:
    mask = np.random.choice([0, 1], size=input.shape, p=[1 - p, p])
    return np.logical_xor(input, mask)


# TER_CODEWORD = {
#     -2: np.array([True, False], dtype=bool),
#     -1: np.array([True, True] , dtype=bool),
#     0: np.array([False, False], dtype=bool),
#     1: np.array([False, True] , dtype=bool),
# }

# No good
# TER_CODEWORD = {
#     1: np.array([True, True] , dtype=bool),
#     0: np.array([True, False], dtype=bool),
#     -1: np.array([False, True], dtype=bool),
#     -2: np.array([False, False] , dtype=bool),
# }

# BEST
TER_CODEWORD = {
    -1: np.array([True, True], dtype=bool),
    -2: np.array([True, False], dtype=bool),
    0: np.array([False, True], dtype=bool),
    1: np.array([False, False], dtype=bool),
}


def inject_bit_flip_module(model: nn.Module, WER: float) -> None:
    for m in model.modules():
        if isinstance(m, BinConv2d) or isinstance(m, BinLinear):
            quant_weight = m.get_quant_weight().sign()
            quant_weight = to_numpy(quant_weight)
            quant_weight = sign_to_bin(quant_weight)

            quant_weight_flip = inject_bit_flip(quant_weight, p=WER)
            quant_weight_flip = bin_to_sign(quant_weight_flip)
            quant_weight_flip = torch.from_numpy(quant_weight_flip).float()
            m.weight = nn.Parameter(quant_weight_flip)
            # m.weight = nn.Parameter(quant_weight)

        elif isinstance(m, TerConv2d) or isinstance(m, TerLinear):
            quant_weight = m.get_quant_weight().sign()
            quant_weight = to_numpy(quant_weight)
            quant_weight = ter_to_bin(quant_weight, TER_CODEWORD)

            quant_weight_flip = inject_bit_flip(quant_weight, p=WER)
            quant_weight_flip = bin_to_ter(quant_weight_flip, TER_CODEWORD)
            quant_weight_flip[quant_weight_flip == -2] = 0
            quant_weight_flip = torch.from_numpy(quant_weight_flip).float()
            m.weight = nn.Parameter(quant_weight_flip)
