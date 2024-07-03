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
from torch.utils.data import Dataset

from research.omtm.tokenizers.base import Tokenizer


class DiscreteIdentity(Tokenizer):
    """Dummy tokenizer for trajectories that are already discrete."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @classmethod
    def create(
        cls, key: str, train_dataset: Dataset, num_classes: int
    ) -> "DiscreteIdentity":
        # add some slack
        return cls(num_classes)

    @property
    def discrete(self) -> bool:
        return True

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        trajectory = torch.nn.functional.one_hot(
            trajectory, num_classes=self.num_classes
        )
        assert trajectory.dim() == 3
        return trajectory.unsqueeze(2).to(torch.float32)

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        assert trajectory.size(2) == 1
        # denormalize trajectory
        trajectory = trajectory.squeeze(2)
        trajectory = torch.argmax(trajectory, dim=-1)
        return trajectory
