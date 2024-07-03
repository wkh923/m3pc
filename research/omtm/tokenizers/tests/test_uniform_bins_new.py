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

import numpy as np
import torch

from research.omtm.tokenizers.uniform_bins import UniformBinningTokenizer


def test_binning_simple():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.rand(1, 2, 3)
    data_min = np.zeros(X.shape[2:])
    data_max = np.ones(X.shape[2:])
    num_bins = 2

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.zeros(data_min.shape), torch.ones(data_max.shape), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_simple_big():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.rand(10000, 2, 3)
    data_min = np.zeros(X.shape[2:])
    data_max = np.ones(X.shape[2:])
    num_bins = 10

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.zeros(data_min.shape), torch.ones(data_max.shape), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_rnd():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.randn(1, 2, 3)
    data_min = np.min(X, axis=(0, 1))
    data_max = np.max(X, axis=(0, 1))
    num_bins = 2

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.tensor(data_min), torch.tensor(data_max), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs + 1e-6)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())


def test_binning_large():
    # generate random data
    rnd = np.random.RandomState(0)

    X = rnd.randn(1024, 8, 5)
    data_min = np.min(X, axis=(0, 1))
    data_max = np.max(X, axis=(0, 1))
    num_bins = 11

    max_diffs = torch.tensor((data_max - data_min) / num_bins / 2)
    tokenizer = UniformBinningTokenizer(
        num_bins, torch.tensor(data_min), torch.tensor(data_max), None
    )
    test_data = torch.tensor(X)

    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    error_np = torch.abs(test_data - recon).numpy()
    error_np = np.max(error_np, (0, 1))
    max_diffs = max_diffs.numpy()
    np.testing.assert_array_less(error_np, max_diffs + 1e-6)

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())
