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

from typing import List

import torch
from torch.utils.data import Dataset

from research.omtm.tokenizers.base import Tokenizer


class ContinuousBinnedTokenizer(Tokenizer):
    """Dummy tokenizer for trajectories that are already discrete."""

    def __init__(self, num_bins, start, end):
        super().__init__()
        self.values = torch.linspace(start, end, steps=num_bins)[None, None, None, :]

    @classmethod
    def create(
        cls,
        key: str,
        train_dataset: Dataset,
        num_bins: int = 64,
        start: float = -1.0,
        end: float = 1.0,
    ) -> "DiscreteIdentity":
        # add some slack
        return cls(num_bins, start, end)

    @property
    def discrete(self) -> bool:
        return True

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3  # B, T, X

        distances = (trajectory[..., None] - self.values.to(trajectory.device)) ** 2
        tokens = torch.argmin(distances, dim=-1)
        # convert to one-hot
        tokens = torch.nn.functional.one_hot(
            tokens, num_classes=self.values.shape[-1]
        ).to(torch.float32)
        assert tokens.dim() == 4  # B, T, X, V
        return tokens

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        assert trajectory.shape[-1] == self.values.shape[-1]

        trajectory = torch.argmax(trajectory, dim=-1)
        return self.values.to(trajectory.device)[0, 0, 0, :][trajectory]
