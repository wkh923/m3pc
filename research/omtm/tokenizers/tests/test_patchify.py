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

from research.omtm.tokenizers.patchify import PatchifyTokenizer


def test_patchify_simple():
    # generate random data
    rnd = np.random.RandomState(0)
    H = 64
    W = 64
    X = np.floor(rnd.rand(1, 2, H, W, 3) * 256)
    tokenizer = PatchifyTokenizer(patch_size=16)
    test_data = torch.tensor(X)
    tokens = tokenizer.encode(test_data)
    recon = tokenizer.decode(tokens)

    ## make sure reconstruction error is less than the max allowed per dimension
    np.testing.assert_allclose(test_data.cpu().numpy(), recon.cpu().numpy())

    ## re-discretize reconstruction and make sure it is the same as original indices
    tokens_2 = tokenizer.encode(recon)
    np.testing.assert_allclose(tokens.cpu().numpy(), tokens_2.cpu().numpy())
