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
from numpy.typing import ArrayLike

from research.omtm.datasets.base import DatasetProtocol, DataStatistics
from research.omtm.models.mtm_model import SquashedNormal
from research.omtm.tokenizers.base import Tokenizer


class ContinuousTokenizer(Tokenizer):
    def __init__(
        self,
        data_mean: ArrayLike,
        data_std: ArrayLike,
        stats: DataStatistics,
        normalize: bool = True,
    ):
        super().__init__()
        self._data_mean = torch.nn.Parameter(
            torch.tensor(data_mean, dtype=torch.float32), requires_grad=False
        )
        self._data_std = torch.nn.Parameter(
            torch.tensor(data_std, dtype=torch.float32), requires_grad=False
        )
        self.stats = stats
        self.normalize = normalize

    @classmethod
    def create(
        cls, key: str, train_dataset: DatasetProtocol, normalize: bool = True
    ) -> "ContinuousTokenizer":
        # TODO: do not normalize action
        data = []
        stats = train_dataset.trajectory_statistics()[key]
        data_mean = stats.mean
        data_std = stats.std
        data_std[data_std < 0.1] = 1  # do not normalize if std is too small
        if key == "actions":
            print("Not normalizing actions")
            return cls(data_mean, data_std, stats, normalize=False)
        return cls(data_mean, data_std, stats, normalize=normalize)

    @property
    def discrete(self) -> bool:
        return False

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3

        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)
            # normalize trajectory
            trajectory = (trajectory - mean) / std
        return trajectory.unsqueeze(2).to(torch.float32)

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(trajectory, SquashedNormal) or trajectory.dim() == 4
        assert isinstance(trajectory, SquashedNormal) or trajectory.size(2) == 1
        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)

            # denormalize trajectory
            return trajectory.squeeze(2) * std + mean
        else:
            return trajectory  # only action don't need .squeeze(2)
