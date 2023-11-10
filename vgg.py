from typing import Tuple

import torch.nn as nn
from torch import Tensor

from pack_norm import PackedBatchNorm1d, PackedBatchNorm2d
from quant_modules import EmbedBinConv2d, EmbedBinLinear, EmbedBinTerConv2d, EmbedBinTerLinear


class VGG13(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l0_0 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.l0_1 = nn.BatchNorm2d(64)
        self.l0_2 = nn.ReLU(inplace=True)

        self.l1_0 = nn.Conv2d(64, 64, 3, bias=False, padding=1)
        self.l1_1 = nn.BatchNorm2d(64)
        self.l1_2 = nn.ReLU(inplace=True)
        self.l1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l2_0 = nn.Conv2d(64, 128, 3, bias=False, padding=1)
        self.l2_1 = nn.BatchNorm2d(128)
        self.l2_2 = nn.ReLU(inplace=True)

        self.l3_0 = nn.Conv2d(128, 128, 3, bias=False, padding=1)
        self.l3_1 = nn.BatchNorm2d(128)
        self.l3_2 = nn.ReLU(inplace=True)
        self.l3_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l4_0 = nn.Conv2d(128, 256, 3, bias=False, padding=1)
        self.l4_1 = nn.BatchNorm2d(256)
        self.l4_2 = nn.ReLU(inplace=True)

        self.l5_0 = nn.Conv2d(256, 256, 3, bias=False, padding=1)
        self.l5_1 = nn.BatchNorm2d(256)
        self.l5_2 = nn.ReLU(inplace=True)
        self.l5_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l6_0 = nn.Conv2d(256, 512, 3, bias=False, padding=1)
        self.l6_1 = nn.BatchNorm2d(512)
        self.l6_2 = nn.ReLU(inplace=True)

        self.l7_0 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.l7_1 = nn.BatchNorm2d(512)
        self.l7_2 = nn.ReLU(inplace=True)
        self.l7_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l8_0 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.l8_1 = nn.BatchNorm2d(512)
        self.l8_2 = nn.ReLU(inplace=True)

        self.l9_0 = nn.Conv2d(512, 512, 3, bias=False, padding=1)
        self.l9_1 = nn.BatchNorm2d(512)
        self.l9_2 = nn.ReLU(inplace=True)
        self.l9_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.l10 = nn.Linear(512, 10, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l0_0(x)
        x = self.l0_1(x)
        x = self.l0_2(x)

        x = self.l1_0(x)
        x = self.l1_1(x)
        x = self.l1_2(x)
        x = self.l1_3(x)

        x = self.l2_0(x)
        x = self.l2_1(x)
        x = self.l2_2(x)

        x = self.l3_0(x)
        x = self.l3_1(x)
        x = self.l3_2(x)
        x = self.l3_3(x)

        x = self.l4_0(x)
        x = self.l4_1(x)
        x = self.l4_2(x)

        x = self.l5_0(x)
        x = self.l5_1(x)
        x = self.l5_2(x)
        x = self.l5_3(x)

        x = self.l6_0(x)
        x = self.l6_1(x)
        x = self.l6_2(x)

        x = self.l7_0(x)
        x = self.l7_1(x)
        x = self.l7_2(x)
        x = self.l7_3(x)

        x = self.l8_0(x)
        x = self.l8_1(x)
        x = self.l8_2(x)

        x = self.l9_0(x)
        x = self.l9_1(x)
        x = self.l9_2(x)
        x = self.l9_3(x)

        x = self.flatten(x)
        x = self.l10(x)
        return x


class EmbedBinVGG13(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l0_0 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.l0_1 = nn.BatchNorm2d(64)
        self.l0_2 = nn.ReLU(inplace=True)

        self.l1_0 = EmbedBinConv2d(64, 64, 3, bias=False, padding=1)
        self.l1_1 = nn.BatchNorm2d(64)
        self.l1_2 = nn.ReLU(inplace=True)
        self.l1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l2_0 = EmbedBinConv2d(64, 128, 3, bias=False, padding=1)
        self.l2_1 = nn.BatchNorm2d(128)
        self.l2_2 = nn.ReLU(inplace=True)

        self.l3_0 = EmbedBinConv2d(128, 128, 3, bias=False, padding=1)
        self.l3_1 = nn.BatchNorm2d(128)
        self.l3_2 = nn.ReLU(inplace=True)
        self.l3_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l4_0 = EmbedBinConv2d(128, 256, 3, bias=False, padding=1)
        self.l4_1 = nn.BatchNorm2d(256)
        self.l4_2 = nn.ReLU(inplace=True)

        self.l5_0 = EmbedBinConv2d(256, 256, 3, bias=False, padding=1)
        self.l5_1 = nn.BatchNorm2d(256)
        self.l5_2 = nn.ReLU(inplace=True)
        self.l5_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l6_0 = EmbedBinConv2d(256, 512, 3, bias=False, padding=1)
        self.l6_1 = nn.BatchNorm2d(512)
        self.l6_2 = nn.ReLU(inplace=True)

        self.l7_0 = EmbedBinConv2d(512, 512, 3, bias=False, padding=1)
        self.l7_1 = nn.BatchNorm2d(512)
        self.l7_2 = nn.ReLU(inplace=True)
        self.l7_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l8_0 = EmbedBinConv2d(512, 512, 3, bias=False, padding=1)
        self.l8_1 = nn.BatchNorm2d(512)
        self.l8_2 = nn.ReLU(inplace=True)

        self.l9_0 = EmbedBinConv2d(512, 512, 3, bias=False, padding=1)
        self.l9_1 = nn.BatchNorm2d(512)
        self.l9_2 = nn.ReLU(inplace=True)
        self.l9_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.l10 = nn.Linear(512, 10, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        x = self.l0_0(x)
        x = self.l0_1(x)
        x0 = self.l0_2(x)

        x = self.l1_0(x0)
        x = self.l1_1(x)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0(x1)
        x = self.l2_1(x)
        x2 = self.l2_2(x)

        x = self.l3_0(x2)
        x = self.l3_1(x)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0(x3)
        x = self.l4_1(x)
        x4 = self.l4_2(x)

        x = self.l5_0(x4)
        x = self.l5_1(x)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0(x5)
        x = self.l6_1(x)
        x6 = self.l6_2(x)

        x = self.l7_0(x6)
        x = self.l7_1(x)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0(x7)
        x = self.l8_1(x)
        x8 = self.l8_2(x)

        x = self.l9_0(x8)
        x = self.l9_1(x)
        x = self.l9_2(x)
        x9 = self.l9_3(x)

        x = self.flatten(x9)
        x10 = self.l10(x)
        return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

    def forward_bin(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        x = self.l0_0(x)
        x = self.l0_1(x)
        x0 = self.l0_2(x)

        x = self.l1_0.forward_bin(x0)
        x = self.l1_1(x)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0.forward_bin(x1)
        x = self.l2_1(x)
        x2 = self.l2_2(x)

        x = self.l3_0.forward_bin(x2)
        x = self.l3_1(x)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0.forward_bin(x3)
        x = self.l4_1(x)
        x4 = self.l4_2(x)

        x = self.l5_0.forward_bin(x4)
        x = self.l5_1(x)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0.forward_bin(x5)
        x = self.l6_1(x)
        x6 = self.l6_2(x)

        x = self.l7_0.forward_bin(x6)
        x = self.l7_1(x)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0.forward_bin(x7)
        x = self.l8_1(x)
        x8 = self.l8_2(x)

        x = self.l9_0.forward_bin(x8)
        x = self.l9_1(x)
        x = self.l9_2(x)
        x9 = self.l9_3(x)

        x = self.flatten(x9)
        x10 = self.l10(x)
        return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)


class PackedEmbedBinVGG13(nn.Module):
    def __init__(self, num_pack: int = 2, mode: str = 'cifar') -> None:
        super().__init__()
        self.mode = mode
        self.l0_0 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.l0_1 = PackedBatchNorm2d(
            num_pack,
            64,
        )
        self.l0_2 = nn.ReLU(inplace=True)

        self.l1_0 = EmbedBinConv2d(64, 64, 3, bias=False, padding=1)
        self.l1_1 = PackedBatchNorm2d(
            num_pack,
            64,
        )
        self.l1_2 = nn.ReLU(inplace=True)
        self.l1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128
        self.l2_0 = EmbedBinConv2d(64, 128, 3, bias=False, padding=1)
        self.l2_1 = PackedBatchNorm2d(
            num_pack,
            128,
        )
        self.l2_2 = nn.ReLU(inplace=True)

        self.l3_0 = EmbedBinConv2d(128, 128, 3, bias=False, padding=1)
        self.l3_1 = PackedBatchNorm2d(
            num_pack,
            128,
        )
        self.l3_2 = nn.ReLU(inplace=True)
        self.l3_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256
        self.l4_0 = EmbedBinConv2d(128, 256, 3, bias=False, padding=1)
        self.l4_1 = PackedBatchNorm2d(
            num_pack,
            256,
        )
        self.l4_2 = nn.ReLU(inplace=True)

        self.l5_0 = EmbedBinConv2d(256, 256, 3, bias=False, padding=1)
        self.l5_1 = PackedBatchNorm2d(
            num_pack,
            256,
        )
        self.l5_2 = nn.ReLU(inplace=True)
        self.l5_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512
        self.l6_0 = EmbedBinConv2d(256, 512, 3, bias=False, padding=1)
        self.l6_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l6_2 = nn.ReLU(inplace=True)

        self.l7_0 = EmbedBinConv2d(512, 512, 3, bias=False, padding=1)
        self.l7_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l7_2 = nn.ReLU(inplace=True)
        self.l7_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l8_0 = EmbedBinConv2d(512, 512, 3, bias=False, padding=1)
        self.l8_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l8_2 = nn.ReLU(inplace=True)

        self.l9_0 = EmbedBinConv2d(512, 512, 3, bias=False, padding=1)
        self.l9_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l9_2 = nn.ReLU(inplace=True)
        self.l9_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        if self.mode == 'cifar':
            self.l10 = nn.Linear(512, 10, bias=True)
        elif self.mode == 'imagenet':
            self.l10_0 = EmbedBinLinear(7 * 7 * 512, 4_096, bias=True)
            self.l10_1 = PackedBatchNorm1d(num_pack, 4_096)
            self.l10_2 = nn.ReLU(inplace=True)

            self.l11_0 = EmbedBinLinear(4_096, 4_096, bias=True)
            self.l11_1 = PackedBatchNorm1d(num_pack, 4_096)
            self.l11_2 = nn.ReLU(inplace=True)

            self.l12 = nn.Linear(4_096, 1_000, bias=True)
        else:
            raise NotImplementedError()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        idx = 0
        x = self.l0_0(x)
        x = self.l0_1(x, idx)
        x0 = self.l0_2(x)

        x = self.l1_0(x0)
        x = self.l1_1(x, idx)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0(x1)
        x = self.l2_1(x, idx)
        x2 = self.l2_2(x)

        x = self.l3_0(x2)
        x = self.l3_1(x, idx)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0(x3)
        x = self.l4_1(x, idx)
        x4 = self.l4_2(x)

        x = self.l5_0(x4)
        x = self.l5_1(x, idx)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0(x5)
        x = self.l6_1(x, idx)
        x6 = self.l6_2(x)

        x = self.l7_0(x6)
        x = self.l7_1(x, idx)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0(x7)
        x = self.l8_1(x, idx)
        x8 = self.l8_2(x)

        x = self.l9_0(x8)
        x = self.l9_1(x, idx)
        x = self.l9_2(x)
        x9 = self.l9_3(x)
        x = self.flatten(x9)

        if self.mode == 'cifar':
            x10 = self.l10(x)
            return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        elif self.mode == 'imagenet':
            x = self.l10_0(x)
            x = self.l10_1(x, idx)
            x10 = self.l10_2(x)

            x = self.l11_0(x10)
            x = self.l11_1(x, idx)
            x11 = self.l11_2(x)

            x12 = self.l12(x11)
            return x12, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
        else:
            raise NotImplementedError()

    def forward_bin(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        idx = 1
        x = self.l0_0(x)
        x = self.l0_1(x, idx)
        x0 = self.l0_2(x)

        x = self.l1_0.forward_bin(x0)
        x = self.l1_1(x, idx)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0.forward_bin(x1)
        x = self.l2_1(x, idx)
        x2 = self.l2_2(x)

        x = self.l3_0.forward_bin(x2)
        x = self.l3_1(x, idx)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0.forward_bin(x3)
        x = self.l4_1(x, idx)
        x4 = self.l4_2(x)

        x = self.l5_0.forward_bin(x4)
        x = self.l5_1(x, idx)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0.forward_bin(x5)
        x = self.l6_1(x, idx)
        x6 = self.l6_2(x)

        x = self.l7_0.forward_bin(x6)
        x = self.l7_1(x, idx)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0.forward_bin(x7)
        x = self.l8_1(x, idx)
        x8 = self.l8_2(x)

        x = self.l9_0.forward_bin(x8)
        x = self.l9_1(x, idx)
        x = self.l9_2(x)
        x9 = self.l9_3(x)

        x = self.flatten(x9)
        if self.mode == 'cifar':
            x10 = self.l10(x)
            return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

        elif self.mode == 'imagenet':
            x = self.l10_0.forward_bin(x)
            x = self.l10_1(x, idx)
            x10 = self.l10_2(x)

            x = self.l11_0.forward_bin(x10)
            x = self.l11_1(x, idx)
            x11 = self.l11_2(x)

            x12 = self.l12(x11)
            return x12, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
        else:
            raise NotImplementedError()


class PackedEmbedBinTerVGG13(nn.Module):
    def __init__(self, num_pack: int = 3, mode: str = 'cifar') -> None:
        super().__init__()
        self.mode = mode
        self.l0_0 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.l0_1 = PackedBatchNorm2d(
            num_pack,
            64,
        )
        self.l0_2 = nn.ReLU(inplace=True)

        self.l1_0 = EmbedBinTerConv2d(64, 64, 3, bias=False, padding=1)
        self.l1_1 = PackedBatchNorm2d(
            num_pack,
            64,
        )
        self.l1_2 = nn.ReLU(inplace=True)
        self.l1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128
        self.l2_0 = EmbedBinTerConv2d(64, 128, 3, bias=False, padding=1)
        self.l2_1 = PackedBatchNorm2d(
            num_pack,
            128,
        )
        self.l2_2 = nn.ReLU(inplace=True)

        self.l3_0 = EmbedBinTerConv2d(128, 128, 3, bias=False, padding=1)
        self.l3_1 = PackedBatchNorm2d(
            num_pack,
            128,
        )
        self.l3_2 = nn.ReLU(inplace=True)
        self.l3_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256
        self.l4_0 = EmbedBinTerConv2d(128, 256, 3, bias=False, padding=1)
        self.l4_1 = PackedBatchNorm2d(
            num_pack,
            256,
        )
        self.l4_2 = nn.ReLU(inplace=True)

        self.l5_0 = EmbedBinTerConv2d(256, 256, 3, bias=False, padding=1)
        self.l5_1 = PackedBatchNorm2d(
            num_pack,
            256,
        )
        self.l5_2 = nn.ReLU(inplace=True)
        self.l5_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512
        self.l6_0 = EmbedBinTerConv2d(256, 512, 3, bias=False, padding=1)
        self.l6_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l6_2 = nn.ReLU(inplace=True)

        self.l7_0 = EmbedBinTerConv2d(512, 512, 3, bias=False, padding=1)
        self.l7_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l7_2 = nn.ReLU(inplace=True)
        self.l7_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l8_0 = EmbedBinTerConv2d(512, 512, 3, bias=False, padding=1)
        self.l8_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l8_2 = nn.ReLU(inplace=True)

        self.l9_0 = EmbedBinTerConv2d(512, 512, 3, bias=False, padding=1)
        self.l9_1 = PackedBatchNorm2d(
            num_pack,
            512,
        )
        self.l9_2 = nn.ReLU(inplace=True)
        self.l9_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        if self.mode == 'cifar':
            self.l10 = nn.Linear(512, 10, bias=True)
        elif self.mode == 'imagenet':
            self.l10_0 = EmbedBinTerLinear(7 * 7 * 512, 4_096, bias=True)
            self.l10_1 = PackedBatchNorm1d(num_pack, 4_096)
            self.l10_2 = nn.ReLU(inplace=True)

            self.l11_0 = EmbedBinTerLinear(4_096, 4_096, bias=True)
            self.l11_1 = PackedBatchNorm1d(num_pack, 4_096)
            self.l11_2 = nn.ReLU(inplace=True)

            self.l12 = nn.Linear(4_096, 1_000, bias=True)
        else:
            raise NotImplementedError()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        idx = 0
        x = self.l0_0(x)
        x = self.l0_1(x, idx)
        x0 = self.l0_2(x)

        x = self.l1_0(x0)
        x = self.l1_1(x, idx)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0(x1)
        x = self.l2_1(x, idx)
        x2 = self.l2_2(x)

        x = self.l3_0(x2)
        x = self.l3_1(x, idx)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0(x3)
        x = self.l4_1(x, idx)
        x4 = self.l4_2(x)

        x = self.l5_0(x4)
        x = self.l5_1(x, idx)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0(x5)
        x = self.l6_1(x, idx)
        x6 = self.l6_2(x)

        x = self.l7_0(x6)
        x = self.l7_1(x, idx)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0(x7)
        x = self.l8_1(x, idx)
        x8 = self.l8_2(x)

        x = self.l9_0(x8)
        x = self.l9_1(x, idx)
        x = self.l9_2(x)
        x9 = self.l9_3(x)
        x = self.flatten(x9)

        if self.mode == 'cifar':
            x10 = self.l10(x)
            return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        elif self.mode == 'imagenet':
            x = self.l10_0(x)
            x = self.l10_1(x, idx)
            x10 = self.l10_2(x)

            x = self.l11_0(x10)
            x = self.l11_1(x, idx)
            x11 = self.l11_2(x)

            x12 = self.l12(x11)
            return x12, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
        else:
            raise NotImplementedError()

    def forward_bin(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        idx = 1
        x = self.l0_0(x)
        x = self.l0_1(x, idx)
        x0 = self.l0_2(x)

        x = self.l1_0.forward_bin(x0)
        x = self.l1_1(x, idx)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0.forward_bin(x1)
        x = self.l2_1(x, idx)
        x2 = self.l2_2(x)

        x = self.l3_0.forward_bin(x2)
        x = self.l3_1(x, idx)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0.forward_bin(x3)
        x = self.l4_1(x, idx)
        x4 = self.l4_2(x)

        x = self.l5_0.forward_bin(x4)
        x = self.l5_1(x, idx)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0.forward_bin(x5)
        x = self.l6_1(x, idx)
        x6 = self.l6_2(x)

        x = self.l7_0.forward_bin(x6)
        x = self.l7_1(x, idx)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0.forward_bin(x7)
        x = self.l8_1(x, idx)
        x8 = self.l8_2(x)

        x = self.l9_0.forward_bin(x8)
        x = self.l9_1(x, idx)
        x = self.l9_2(x)
        x9 = self.l9_3(x)

        x = self.flatten(x9)
        if self.mode == 'cifar':
            x10 = self.l10(x)
            return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

        elif self.mode == 'imagenet':
            x = self.l10_0.forward_bin(x)
            x = self.l10_1(x, idx)
            x10 = self.l10_2(x)

            x = self.l11_0.forward_bin(x10)
            x = self.l11_1(x, idx)
            x11 = self.l11_2(x)

            x12 = self.l12(x11)
            return x12, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
        else:
            raise NotImplementedError()

    def forward_ter(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        idx = 2
        x = self.l0_0(x)
        x = self.l0_1(x, idx)
        x0 = self.l0_2(x)

        x = self.l1_0.forward_ter(x0)
        x = self.l1_1(x, idx)
        x = self.l1_2(x)
        x1 = self.l1_3(x)

        x = self.l2_0.forward_ter(x1)
        x = self.l2_1(x, idx)
        x2 = self.l2_2(x)

        x = self.l3_0.forward_ter(x2)
        x = self.l3_1(x, idx)
        x = self.l3_2(x)
        x3 = self.l3_3(x)

        x = self.l4_0.forward_ter(x3)
        x = self.l4_1(x, idx)
        x4 = self.l4_2(x)

        x = self.l5_0.forward_ter(x4)
        x = self.l5_1(x, idx)
        x = self.l5_2(x)
        x5 = self.l5_3(x)

        x = self.l6_0.forward_ter(x5)
        x = self.l6_1(x, idx)
        x6 = self.l6_2(x)

        x = self.l7_0.forward_ter(x6)
        x = self.l7_1(x, idx)
        x = self.l7_2(x)
        x7 = self.l7_3(x)

        x = self.l8_0.forward_ter(x7)
        x = self.l8_1(x, idx)
        x8 = self.l8_2(x)

        x = self.l9_0.forward_ter(x8)
        x = self.l9_1(x, idx)
        x = self.l9_2(x)
        x9 = self.l9_3(x)

        x = self.flatten(x9)
        if self.mode == 'cifar':
            x10 = self.l10(x)
            return x10, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

        elif self.mode == 'imagenet':
            x = self.l10_0.forward_ter(x)
            x = self.l10_1(x, idx)
            x10 = self.l10_2(x)

            x = self.l11_0.forward_ter(x10)
            x = self.l11_1(x, idx)
            x11 = self.l11_2(x)

            x12 = self.l12(x11)
            return x12, (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    import torch

    input = torch.randn(2, 3, 32, 32)
    model = VGG13()
    model = model.eval()
    output = model(input)
    print(output.shape)

    model = EmbedBinVGG13()
    model = model.eval()
    output = model(input)
    print(output.shape)

    output = model.forward_bin(input)
