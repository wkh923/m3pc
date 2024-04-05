# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
