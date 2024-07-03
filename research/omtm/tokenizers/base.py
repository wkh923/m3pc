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

from abc import ABC, abstractclassmethod
from typing import Dict, Type, TypeVar

import torch
from torch.utils.data import Dataset

T = TypeVar("T", bound="Tokenizer")


class Tokenizer(torch.nn.Module, ABC):
    @abstractclassmethod
    def create(cls: Type[T], key: str, train_dataset: Dataset, **kwargs) -> T:
        """Create a new instance of the model."""

    @property
    def discrete(self) -> bool:
        """Whether the tokenizer is discrete or continuous."""

    def encode(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode a trajectory.

        Args:
            trajectories (torch.Tensor): shape=(batch_size, L, ...))

        Returns:
            tokenized_trajectories (torch.Tensor): shape=(batch_size, L, tokens_per_dim, tokens_feature_size)
        """
        ...

    def decode(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Decode a trajectory.

        Args:
            tokenized_trajectories (torch.Tensor): shape=(batch_size, L, tokens_per_dim, tokens_feature_size)

        Returns:
            trajectories (torch.Tensor): shape=(batch_size, L, ...))
        """
        ...


class TokenizerManager(torch.nn.Module):
    def __init__(self, tokenizers: Dict[str, Tokenizer]):
        super().__init__()
        self.tokenizers = torch.nn.ModuleDict(tokenizers)

    def encode(self, trajectories: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode all of the trajectories.

        Args:
            trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, ...)). This could be states, actions, images, etc

        Returns:
            tokenized_trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, tokens_per_dim, tokens_feature_size))
        """
        out_trajectories = {}
        for key, value in trajectories.items():
            if key in self.tokenizers.keys():
                out_trajectories[key] = self.tokenizers[key].encode(value)
                assert len(out_trajectories[key].shape) == 4
        return out_trajectories

    def decode(
        self, tokenzied_trajectories: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Decode all of the trajectories.

        Args:
            tokenized_trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, tokens_per_dim, tokens_feature_size))

        Returns:
            trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, ...)).
        """
        out_trajectories = {}
        for key, value in tokenzied_trajectories.items():
            out_trajectories[key] = self.tokenizers[key].decode(value)
        return out_trajectories
