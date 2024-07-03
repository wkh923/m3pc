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

from research.omtm.tokenizers.continuous_binned import ContinuousBinnedTokenizer


def test_binning_simple():
    # generate random data
    rnd = np.random.RandomState(0)

    X = [
        [0, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0],
    ]
    X = np.array(X)[None]
    tokenizer = ContinuousBinnedTokenizer([0, 0.1, 0.2])

    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    np.testing.assert_allclose(recon, test_data)


def test_binning_simple_with_others():
    # generate random data
    rnd = np.random.RandomState(0)

    X = [
        [0, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0],
    ]
    X = np.array(X)[None]
    tokenizer = ContinuousBinnedTokenizer([-0.1, 0, 0.1, 0.2, 0.4, 0.5])

    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    np.testing.assert_allclose(recon, test_data)


def test_binning_simple_logits():
    # generate random data
    rnd = np.random.RandomState(0)

    X = [
        [0, 0, 0.1],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0],
    ]
    X = np.array(X)[None]
    tokenizer = ContinuousBinnedTokenizer([-0.1, 0, 0.1, 0.2, 0.4, 0.5])

    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens * 100)

    np.testing.assert_allclose(recon, test_data)
