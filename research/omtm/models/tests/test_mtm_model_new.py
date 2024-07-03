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

from research.omtm.models.omtm_model import omtm, omtmConfig


def test_maskdp_model_simple():
    # smoke test to check that the model can run
    features_dim = 13
    action_dim = 7
    n_layer = 1
    dropout = 0.0
    traj_length = 20

    data_shapes = {
        "actions": (3, action_dim),
        "states": (1, features_dim),
    }

    model = omtm(
        data_shapes,
        traj_length,
        omtmConfig(
            n_embd=128,
            n_head=2,
            n_enc_layer=n_layer,
            n_dec_layer=n_layer,
            dropout=dropout,
        ),
    )

    trajectories_torch = {
        "actions": torch.randn(5, traj_length, *data_shapes["actions"]),
        "states": torch.randn(5, traj_length, *data_shapes["states"]),
    }
    masks = {
        "actions": torch.ones(traj_length, 3),
        "states": torch.ones(traj_length, 1),
    }

    out_trajs = model(trajectories_torch, masks)
    for k, v in trajectories_torch.items():
        assert v.shape == out_trajs[k].shape
