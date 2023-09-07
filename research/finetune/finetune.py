import os
import pprint
import random
import time
import wandb
from collections import defaultdict
from dataclasses import dataclass, replace, field
from typing import Any, Callable, Dict, Sequence, Tuple, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig, OmegaConf


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
from research.finetune.model import Critic

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
    
    pretrain_discount: float = 1.5
    
    
    discount: float = 0.99
    """Discount factor"""
    
    policy_std: float = 1.0
    """Policy noise std when planning with CEM"""
    
    num_updates_per_online_rollout: int = 300
    
    n_iter: int = 5
    """The number of CEM iterations"""
    
    n_rsamples: int = 800
    """The number of random action samples"""
    
    n_policy_samples: int = 100
    """The number of policy action samples"""
    
    top_k: int = 100
    """The number of action samples used for CEM update"""
    
    use_masked_loss: bool = True
    
    loss_weight: Dict[str, float] = field(default_factory=lambda: {
        "actions": 1.0, "states": 1.0, "returns": 1.0,
        "rewards": 1.0, "policy": 1.0
    })
    
    clip_min: float = -1.0
    
    clip_max: float = 1.0
    
    action_noise_std: float = 0.2
    
    tau : float = 0.1
    
    critic_hidden_size: int = 256


def main(hydra_cfg):
    dp: DistributedParams = get_distributed_params()
    torch.cuda.set_device(dp.local_rank)
    
    pretrain_model_path = hydra_cfg.pretrain_model_path
    pretrain_critic_path = hydra_cfg.pretrain_critic_path
    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.finetune_args)
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    cfg.traj_length = hydra_cfg.pretrain_args.traj_length
    cfg.env_name = hydra_cfg.pretrain_args.env_name
    env = gym.make(cfg.env_name)
    
    
    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)
    
    print(os.getcwd())
    logger.info(f"Working directory: {os.getcwd()}")
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))
        
    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol
    
    train_dataset, val_dataset, train_raw_dataset = hydra.utils.call(
        hydra_cfg.pretrain_dataset, seq_steps=cfg.traj_length
    )
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = DataLoader(
            val_dataset,
            # shuffle=False,
            batch_size=cfg.batch_size,
            num_workers=1,
            sampler=val_sampler,
        )
    
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
    
    buffer = ReplayBuffer(train_raw_dataset, cfg.pretrain_discount, cfg.traj_length, batch_size=cfg.batch_size, 
                          buffer_size=cfg.buffer_size, name=cfg.env_name, mode="prob")
    dataloader = iter(buffer)
    batch_example = next(dataloader)
    batch_example = {k: v.to(cfg.device, non_blocking=True) for k, v in batch_example.items()}
    
    data_shapes = {}
    for k, v in tokenizer_manager.encode(batch_example).items():
        data_shapes[k] = v.shape[-2:]
    
    print(f"Data shapes: {data_shapes}")
    logger.info(f"Data shapes: {data_shapes}")
    
    learner = Learner(cfg, env, data_shapes, model_config, pretrain_model_path, pretrain_critic_path, tokenizer_manager, discrete_map)
    
    # create a wandb logger and log params of interest
    wandb_cfg_log_dict = OmegaConf.to_container(hydra_cfg)
    wandb_cfg_log_dict["*discrete_map"] = discrete_map
    wandb_cfg_log_dict["*path"] = str(os.getcwd())
    wandb_cfg_log_dict["*git_hash"] = get_git_hash()
    wandb_cfg_log_dict["*git_dirty"] = get_git_dirty()
    wandb_cfg_log_dict["*hydra_cfg_hash"] = get_cfg_hash(hydra_cfg)
    wandb_cfg_log_dict["*num_parameters"] = sum(
        p.numel() for p in learner.mtm.parameters() if p.requires_grad
    )
    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=f"{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity or None,
        resume=hydra_cfg.wandb.resume,
        group=dp.job_id,
    )
    
    if wandb_cfg_log.resume:
        exp_id = wandb_cfg_log_dict["*hydra_cfg_hash"]
        wandb_cfg_log = replace(
            wandb_cfg_log,
            experiment_id=exp_id,
        )
    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)
    
    step = 0
    if wandb_logger.enable and wandb.run.resumed:
        logger.info("Trying to resume ...")
        ckpt_mtm_path = get_ckpt_path_from_folder(
            os.getcwd(), "mtm"
        )
        ckpt_critic_path = get_ckpt_path_from_folder(
            os.getcwd(), "critic"
        )
        if ckpt_mtm_path is not None and ckpt_critic_path is not None:
            ckpt = torch.load(ckpt_mtm_path, map_location=cfg.device)
            logger.info(f"Resuming from checkpoint: {ckpt_mtm_path}")
            step_mtm = ckpt["step"]
            learner.mtm.load_state_dict(ckpt["model"])
            learner.mtm_optimizer.load_state_dict(ckpt["optimizer"])
    
            ckpt = torch.load(ckpt_critic_path, map_location=cfg.device)
            logger.info(f"Resuming from checkpoint: {ckpt_critic_path}")
            step_critic = ckpt["step"]
            learner.critic.load_state_dict(ckpt["model"])
            learner.critic_optimizer.load_state_dict(ckpt["optimizer"])
            learner.critic_target.load_state_dict(ckpt["model"])
            
            step = max(step_mtm, step_critic)
        else:
            logger.info(f"No checkpoints found, starting from scratch.")
    logger.info(f"starting from step={step}")
    
    epoch = 0
    while True:
        B = time.time()
        log_dict = {}
        log_dict["train/epochs"] = epoch
        
        start_time = time.time()
        try:
            batch = next(dataloader)
        except StopIteration:
            buffer.online_rollout(learner.mtm, tokenizer_manager, cfg.device, num_trajectories=cfg.num_updates_per_online_rollout, percentage=1, 
                                  clip_min=cfg.clip_min, clip_max=cfg.clip_max, action_noise_std=cfg.action_noise_std)
            dataloader = iter(buffer)
            batch = next(dataloader)    
        
        batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
        mtm_log = learner.mtm_update(batch)
        critic_log = learner.critic_update(batch)
        log_dict.update(mtm_log)
        log_dict.update(critic_log)
        log_dict["time/train_step"] = time.time() - start_time
        
        learner.critic_target_soft_update()
        
        if step % cfg.print_every == 0:
            try:
                train_loss = log_dict["train/loss"]
            except:
                train_loss = -1
            logger.info(f"Step: {step}, Train Loss: {train_loss}")
            
        if dp.rank == 0 and step % cfg.save_every == 0:
            torch.save(
                {
                    "model": learner.mtm.state_dict(),
                    "optimizer": learner.mtm_optimizer.state_dict(),
                    "step": step,
                },
                f"mtm_{step}.pt"
            )
            torch.save(
                {
                    "model": learner.critic.state_dict(),
                    "optimizer": learner.critic_optimizer.state_dict(),
                    "step": step,
                },
                f"critic_{step}.pt"
            )
            try:
                if step > 3 * cfg.save_every:
                    remove_step = step - 3 * cfg.save_every
                    if (remove_step // cfg.save_every) % 10 != 0:
                        os.remove(f"mtm_{remove_step}.pt")
                        os.remove(f"critic_{remove_step}.pt")
            except Exception as e:
                logger.error(f"Failed to remove model file! {e}")
            
        if step % cfg.eval_every == 0:
            start_time = time.time()
            learner.mtm.eval()
            learner.critic.eval()
            
            val_batch = next(iter(val_loader))
            val_batch = {
                k: v.to(cfg.device, non_blocking=True) for k, v in val_batch.items()
            }
            (
                loss,
                losses,
                masked_losses,
                masked_c_losses
            ) = learner.compute_mtm_loss(val_batch)
            
            log_dict["eval/loss"] = loss.item()
            for k, v in losses.items():
                log_dict[f"eval/loss_{k}"] = v
            for k, v in masked_losses.items():
                log_dict[f"eval/masked_loss_{k}"] = v
            for k, v in masked_c_losses.items():
                log_dict[f"eval/masked_c_loss_{k}"] = v
            
            val_dict = learner.evaluate(num_episodes=10)
            
            log_dict.update(val_dict)
            learner.mtm.train()
            learner.critic.train()
            log_dict["time/eval_step_time"] = time.time() - start_time
        
        log_dict["time/iteration_step_time"] = time.time() - B
        
        if random.randint(0, cfg.log_every) == 0:
            logger.info(f"Step {step}")
            wandb_logger.log(log_dict, step=step)
        
        step += 1
        if step >= cfg.num_train_steps:
            break
    
    torch.save(
                {
                    "model": learner.mtm.state_dict(),
                    "optimizer": learner.mtm_optimizer.state_dict(),
                    "step": step,
                },
                f"mtm_{step}.pt"
            )
    torch.save(
        {
            "model": learner.critic.state_dict(),
            "optimizer": learner.critic_optimizer.state_dict(),
            "step": step,
        },
        f"critic_{step}.pt"
    )
    
    

@hydra.main(config_path=".", config_name="config", version_base = "1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)
  


if __name__ == "__main__":
    configure_jobs()