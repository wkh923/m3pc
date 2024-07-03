# MIT License

# Copyright (c) 2023 Meta Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from research.omtm.datasets.base import DatasetProtocol
from research.omtm.tokenizers.base import Tokenizer


def extract_patches(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, H, W, C = inputs.shape

    assert H % patch_size == 0
    assert W % patch_size == 0
    P_H = H // patch_size
    P_W = W // patch_size
    x = inputs.reshape(B, P_H, patch_size, P_W, patch_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, P_H * P_W, patch_size**2 * C)
    return x


def merge_patches(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, L, _ = inputs.shape
    H = W = int(L**0.5)
    x = inputs.reshape(B, H, W, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, H * patch_size, W * patch_size, -1)
    return x


class PatchifyTokenizer(Tokenizer):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    @classmethod
    def create(
        cls, key: str, train_dataset: DatasetProtocol, patch_size: int
    ) -> "PatchifyTokenizer":
        return cls(patch_size)

    @property
    def discrete(self) -> bool:
        return False

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # check shape is consistant with images
        assert trajectory.dim() == 5
        assert trajectory.min() >= 0
        assert trajectory.max() <= 255

        # normalize trajectory
        trajectory = (trajectory / 255) - 0.5

        # extract patches
        # reshape to (B, L, H, W, C)
        B, L, H, W, C = trajectory.shape
        trajectory = trajectory.reshape(B * L, H, W, C)
        patches = extract_patches(trajectory, self.patch_size)
        patches = patches.reshape(B, L, -1, self.patch_size**2 * C)
        return patches

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        # check shape is consistant with patches
        assert trajectory.dim() == 4
        # trajectory shape, (B, L, P, C)
        B, L, P, C = trajectory.shape
        # reshape
        trajectory = trajectory.reshape(B * L, P, C)
        trajectory = merge_patches(trajectory, self.patch_size)
        trajectory = trajectory.reshape(B, L, *trajectory.shape[1:])
        # denormalize trajectory
        trajectory = ((trajectory + 0.5) * 255).round()
        return torch.clamp(trajectory, 0, 255)
