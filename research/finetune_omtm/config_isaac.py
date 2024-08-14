import os
import pprint
import random
import time
import wandb
from collections import defaultdict
from dataclasses import dataclass, replace, field
from typing import Any, Callable, Dict, Sequence, Tuple, List, Optional
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig, OmegaConf



# from research.jaxrl.utils import make_env
from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from research.omtm.datasets.base import DatasetProtocol
from research.omtm.distributed_utils import DistributedParams, get_distributed_params
from research.omtm.tokenizers.base import Tokenizer, TokenizerManager
from research.omtm.utils import (
    get_cfg_hash,
    get_ckpt_path_from_folder,
    get_git_dirty,
    get_git_hash,
    set_seed_everywhere,
)



from research.finetune_omtm.replay_buffer_isaac import ReplayBuffer
from research.omtm.datasets.isaac_sim_wrapper import MTMEnvWrapper
from research.omtm.datasets.isaac_dataset import ISAACDataset
from research.finetune_omtm.learner import Learner

@dataclass
class RunConfig:
    seed: int = 0
    """RNG seed."""

    traj_batch_size: int = 64
    """Batch size used during MTM training."""

    traj_buffer_size: int = 10000
    """Max trajectory level replay buffer size"""

    trans_batch_size: int = 256
    """"Batch size used during critic training"""

    trans_buffer_size: int = 20000
    """"Max transition level replay buffer size"""

    using_online_threshold: int = 5000

    buffer_init_ratio: float = 0.2

    log_every: int = 100
    """Print training loss every N steps."""

    print_every: int = 1000
    """Print training loss every N steps."""

    eval_every: int = 5000
    """Evaluate model every N steps."""

    save_every: int = 5000
    """Evaluate model every N steps."""

    device: str = "cuda:0"
    """Device to use for training."""

    warmup_steps: int = 1_000
    """Number of warmup steps for learning rate scheduler."""

    num_train_steps: int = 5_000_000
    """Number of training steps."""

    explore_steps: int = 1_000_000

    learning_rate: float = 1e-3
    """Learning rate for MTM"""

    critic_lr: float = 5e-4
    """"Learning rate for Q"""

    v_lr: float = 5e-4
    """"Learning rate for V"""

    weight_decay: float = 1e-5
    """Weight decay."""

    traj_length: int = 1
    """Trajectory length."""

    env_name: str = "Hopper-v2"
    """Gym environment"""

    pretrain_discount: float = 1.5
    "Discount in pretrain phase which is used to calculate RTG"

    discount: float = 0.99
    """Discount factor in finetuning phase"""

    mtm_iter_per_rollout: int = 300
    """MTM iterations per online rollout"""

    v_iter_per_mtm: int = 10
    """"V update iterations per MTM update"""

    action_samples: int = 10
    """The number of policy action samples"""

    clip_min: float = -1.0
    """"Minimum action"""

    clip_max: float = 1.0
    """"Maximum action"""

    temperature: float = 0.5
    """Used in planning"""

    action_noise_std: float = 0.2

    tau: float = 0.1
    """Used to construct replay buffer"""

    select_mode: str = "uniform"
    """uniform: sample the trajactories randomly;
    prob: assign higher sample probability to trajectories with higher return"""

    mask_ratio: Tuple = (0.5,)
    """The ratio of masked token during MTM training phase"""

    p_weights: Tuple = (0.1, 0.1, 0.7, 0.1)
    """The probability of different madalities to be autoregressive"""

    critic_hidden_size: int = 256
    """"Hidden size of Q and V"""

    plan: bool = True
    """Do planning when rolling out"""

    plan_guidance: str = "mtm_critic"
    lmbda: float = 0.9
    expectile: float = 0.8

    horizon: int = 4
    """The horizon for planning, horizon=1 means critic guided search"""
    rtg_percent: float = 1.0
    
    index_jump: int = 0
