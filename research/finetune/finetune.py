import os
import pprint
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Sequence, Tuple, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
import wandb
from omegaconf import DictConfig, OmegaConf
import torch.dis



from research.jaxrl.utils import make_env
from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.datasets.sequence_dataset import Trajectory
from research.mtm.distributed_utils import DistributedParams, get_distributed_params
from research.mtm.models.mtm_model import MTM, make_plots_with_masks
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.mtm.utils import (
    get_cfg_hash,
    get_ckpt_path_from_folder,
    get_git_dirty,
    get_git_hash,
    set_seed_everywhere,
)
from research.finetune.replay_buffer import ReplayBuffer
from research.finetune.learner import Learner

import gym


@dataclass
class RunConfig:
    seed: int = 0
    """RNG seed."""

    batch_size: int = 64
    """Batch size used during training."""
    
    buffer_size: int = 10000
    """Max replay buffer size"""

    log_every: int = 100
    """Print training loss every N steps."""

    print_every: int = 1000
    """Print training loss every N steps."""

    eval_every: int = 5000
    """Evaluate model every N steps."""

    save_every: int = 5000
    """Evaluate model every N steps."""

    device: str = "cuda"
    """Device to use for training."""

    warmup_steps: int = 1_000
    """Number of warmup steps for learning rate scheduler."""

    num_train_steps: int = 5_000_000
    """Number of training steps."""

    learning_rate: float = 1e-3
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay."""

    traj_length: int = 1
    """Trajectory length."""
    
    env_name: str = "Hopper-v2"
    """Gym environment"""
    
    discount: int = 0.99
    """Discount factor"""
    
    policy_std: float = 1.0
    """Policy noise std when planning with CEM"""
    
    n_iter: int = 5
    """The number of CEM iterations"""
    
    n_rsamples: int = 800
    """The number of random action samples"""
    
    n_policy_samples: int = 100
    """The number of policy action samples"""
    
    top_k: int = 100
    """The number of action samples used for CEM update"""
    
    use_masked_loss: bool = True
    
    loss_weight: Dict[str, float] = {"actions":  1.0,"states": 1.0, "returns": 1.0, "rewards": 1.0, "policy": 1.0}
    """Loss weight for MTM model"""
    
    clip_min: float = -1.0
    
    clip_max: float = 1.0
    
    action_noise_std: float = 0.2
    
    tau : float = 0.1


def main(hydra_cfg):
    pretrain_model_path = hydra_cfg.pretrain_model_path
    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.finetune_args)
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    cfg.traj_length = hydra_cfg.pretrain_args.traj_length
    cfg.env_name = hydra_cfg.pretrain_args.env_name
    
    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)
    
    print(os.getcwd())
    logger.info(f"Working directory: {os.getcwd()}")
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))
        
    train_dataset: DatasetProtocol
    
    train_dataset, _ = hydra.utils.call(
        hydra_cfg.pretrain_dataset, seq_steps=cfg.traj_length
    )
    print(len(train_dataset))
    
    if "tokenizers" in hydra_cfg:
        tokenizers: Dict[str, Tokenizer] = {
            k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
            for k, v in hydra_cfg.tokenizers.items()
        }
    tokenizer_manager = TokenizerManager(tokenizers).to(cfg.device)
    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete
    logger.info(f"Tokenizers: {tokenizers}")
    
    env = make_env(cfg.env_name, cfg.seed)
    assert discrete_map["actions"] == isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete_map["states"] == isinstance(env.observation_space, gym.spaces.Discrete)
    action_shape = env.action_space.shape if not discrete_map["actions"] else (1,)
    observation_shape = env.observation_space.shape if not discrete_map["states"] else (1,)
    
    buffer = ReplayBuffer(train_dataset, cfg.discount, cfg.traj_length, batch_size=cfg.batch_size, 
                          buffer_size=cfg.buffer_size, name=cfg.env_name, mode="prob")
    dataloader = iter(buffer)
    batch_example = next(dataloader)
    
    data_shapes = {}
    for k, v in tokenizer_manager.encode(batch_example).items():
        data_shapes[k] = v.shape[-2:]
    
    print(f"Data shapes: {data_shapes}")
    logger.info(f"Data shapes: {data_shapes}")
    
    pretrain_model = model_config.create(data_shapes, cfg.traj_length)
    pretrain_model.load_state_dict(torch.load(pretrain_model_path + "model_60000.pt")["model"])
    pretrain_model.to(cfg.device)
    
    learner = Learner(cfg, pretrain_model, critic_model, tokenizer_manager, discrete_map)
    
    
    
    

@hydra.main(config_path=".", config_name="config", version_base = "1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)
  


if __name__ == "__main__":
    configure_jobs()