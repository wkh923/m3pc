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

import hashlib
import os
import random
from typing import Optional

import git
import numpy as np
import torch
from omegaconf import OmegaConf

from research.omtm.masks import MaskType


def load_hydra_path(path):
    hydra_cfg = OmegaConf.load(os.path.join(path, ".hydra/config.yaml"))

    # deal with mask_indicies -> mask_pattern change
    mask_indicies = hydra_cfg.args.mask_indicies
    del hydra_cfg.args.mask_indices
    mask_pattern_names = [member.name for member in MaskType]
    mask_pattern = mask_pattern_names[mask_indicies]
    hydra_cfg.args.mask_patterns = mask_pattern
    return hydra_cfg


def get_ckpt_path_from_folder(folder, model: str = "model") -> Optional[str]:
    steps = []
    names = []
    paths_ = os.listdir(folder)
    for name in [os.path.join(folder, n) for n in paths_ if "pt" in n and model in n]:
        step = os.path.basename(name).split("_")[-1].split(".")[0]
        steps.append(step)
        names.append(name)

    if len(steps) == 0:
        return None
    else:
        ckpt_path = names[np.argmax(steps)]
        return ckpt_path


def get_cfg_hash(hydra_cfg):
    m = hashlib.md5()
    m.update(OmegaConf.to_yaml(hydra_cfg).encode("utf-8"))
    return m.hexdigest()


def get_git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return str(sha)


def get_git_dirty() -> bool:
    repo = git.Repo(search_parent_directories=True)
    return repo.is_dirty()


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
