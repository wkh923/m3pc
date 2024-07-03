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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

import numpy as np

from research.omtm.tokenizers.base import TokenizerManager


@dataclass
class DataStatistics:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
        self.min = np.array(self.min, dtype=np.float32)
        self.max = np.array(self.max, dtype=np.float32)

        # check shapes
        assert self.mean.shape == self.std.shape == self.min.shape == self.max.shape

        # check ordering
        assert np.all(self.min <= self.max)


class DatasetProtocol(Protocol):
    @property
    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        ...
        """Shapes of the trajectories in the dataset."""

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Returns the observation, action, and return for the given index.

        Args:
            idx: The index of the data point to return.

        Returns:
            trajectories: A dictionary of trajectories.
        """

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        """Returns the evaluation logs for the given model.

        Args:
            model: The model to evaluate.

        Returns:
            logs: A dictionary of evaluation logs.
        """
