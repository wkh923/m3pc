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

from research.jaxrl.utils import make_env
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
from research.finetune_omtm.replay_buffer import ReplayBuffer
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

    device: str = "cuda"
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


def main(hydra_cfg):
    dp: DistributedParams = get_distributed_params()
    torch.cuda.set_device(hydra_cfg.local_cuda_rank)

    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.finetune_args)
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    cfg.traj_length = hydra_cfg.pretrain_args.traj_length
    cfg.env_name = hydra_cfg.pretrain_args.env_name
    

    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)

    current_path = os.getcwd()
    print(current_path)
    logger.info(f"Working directory: {current_path}")
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))

    pretrain_model_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.pretrain_model_path)
    )
    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol

    train_dataset, val_dataset, train_raw_dataset = hydra.utils.call(
        hydra_cfg.pretrain_dataset, seq_steps=cfg.traj_length
    )
    env = make_env(cfg.env_name, cfg.seed)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        # shuffle=False,
        batch_size=cfg.traj_batch_size,
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

    buffer = ReplayBuffer(cfg, train_raw_dataset, cfg.pretrain_discount)
    obs_mean = torch.tensor(buffer.obs_mean, device=cfg.device)
    obs_std = torch.tensor(buffer.obs_std, device=cfg.device)
    dataloader = iter(buffer)
    batch_example = next(dataloader)

    data_shapes = {}
    for k, v in tokenizer_manager.encode(batch_example).items():
        data_shapes[k] = v.shape[-2:]

    print(f"Data shapes: {data_shapes}")
    logger.info(f"Data shapes: {data_shapes}")

    learner = Learner(
        cfg,
        env,
        data_shapes,
        model_config,
        pretrain_model_path,
        obs_mean,
        obs_std,
        tokenizer_manager,
        discrete_map,
    )

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
    # change to hours minute and second

    current_time = datetime.now().strftime("%m%d_%H")
    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=hydra_cfg.wandb.experiment_name
        + "_"
        + current_time
        + f"{dp.job_id}",
        # experiment_id=f"{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity or None,
        resume=hydra_cfg.wandb.resume,
        group=hydra_cfg.wandb.group,
    )

    if wandb_cfg_log.resume:
        exp_id = wandb_cfg_log_dict["*hydra_cfg_hash"]
        wandb_cfg_log = replace(
            wandb_cfg_log,
            experiment_id=exp_id,
        )
    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)

    if cfg.warmup_steps > 0:

        for i in range(cfg.warmup_steps):
            batch = buffer.trans_sample()
            critic_log = learner.critic_update(batch)

            if i % 5000 == 0:

                learner.iql.actor.eval()
                _, eval_score = learner.evaluate_policy(num_episodes=10)
                learner.iql.actor.train()

                print("---------------------------------------")
                print(
                    f"IQL Evaluation over {i} timesteps: "
                    f"D4RL score: {eval_score:.3f}"
                )
                print("---------------------------------------")

    step = 0

    logger.info(f"starting from step={step}")

    episode = 0
    while True:
        B = time.time()
        log_dict = {}
        log_dict["train/episodes"] = episode

        start_time = time.time()

        for critic_iter in range(cfg.v_iter_per_mtm):
            experiences = buffer.trans_sample()
            critic_log = learner.critic_update(experiences)

        log_dict.update(critic_log)

        try:
            batch = next(dataloader)
        except StopIteration:
            logger.info(f"Rollout a new trajectory")
            learner.mtm.eval()
            explore_dict = buffer.online_rollout(learner.action_sample)
            learner.mtm.train()
            episode += 1
            dataloader = iter(buffer)
            batch = next(dataloader)
            wandb_logger.log(explore_dict, step=step)

        mtm_log = learner.mtm_update(batch, data_shapes, discrete_map)
        log_dict.update(mtm_log)

        log_dict["time/train_step"] = time.time() - start_time
        log_dict["time/online_step"] = buffer.total_step

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
                f"mtm_{step}.pt",
            )
            torch.save(learner.iql.state_dict(), f"iql_{step}.pt")

            try:
                if step > 3 * cfg.save_every:
                    remove_step = step - 3 * cfg.save_every
                    if (remove_step // cfg.save_every) % 10 != 0:
                        os.remove(f"mtm_{remove_step}.pt")
                        os.remove(f"iql_{remove_step}.pt")
            except Exception as e:
                logger.error(f"Failed to remove model file! {e}")

        if step % cfg.eval_every == 0:
            start_time = time.time()
            learner.mtm.eval()
            learner.iql.qf.eval()
            learner.iql.vf.eval()
            learner.iql.actor.eval()

            val_batch = next(iter(val_loader))
            val_batch = {
                k: v.to(cfg.device, non_blocking=True) for k, v in val_batch.items()
            }
            (loss, losses, masked_losses, masked_c_losses, entropy) = (
                learner.compute_mtm_loss(
                    val_batch,
                    data_shapes,
                    discrete_map,
                    learner.mtm.temperature().detach(),
                )
            )

            log_dict["eval/loss"] = loss.item()
            for k, v in losses.items():
                log_dict[f"eval/loss_{k}"] = v
            for k, v in masked_losses.items():
                log_dict[f"eval/masked_loss_{k}"] = v
            for k, v in masked_c_losses.items():
                log_dict[f"eval/masked_c_loss_{k}"] = v
            log_dict[f"eval/entropy"] = entropy.item()

            val_dict, _ = learner.evaluate(
                num_episodes=10, episode_rtg_ref=buffer.values_up_bound
            )
            if cfg.plan is True:
                plan_dict, _ = learner.evaluate_plan(
                    num_episodes=5, episode_rtg_ref=buffer.values_up_bound
                )
            val_dict.update(plan_dict)
            # iql_dict, _ = learner.evaluate_policy(num_episodes=10)
            # val_dict.update(iql_dict)

            learner.mtm.train()
            learner.iql.qf.train()
            learner.iql.vf.train()
            learner.iql.actor.train()
            # val_dict["time/eval_step_time"] = time.time() - start_time

            if "hopper" in cfg.env_name:
                return_max = 4000
                step_max = 1000
            elif "walker2d" in cfg.env_name:
                return_max = 5000
                step_max = 1000
            elif "halfcheetah" in cfg.env_name:
                return_max = 12000
                step_max = 1000
            else:
                raise NotImplementedError
            explore_return_hist = np.histogram(
                buffer.p_return_list, bins=list(range(0, return_max + 1, 50))
            )
            explore_length_hist = np.histogram(
                buffer.p_length_list, bins=list(range(0, step_max + 1, 20))
            )

            log_dict["explore/explore_return_hist"] = wandb.Histogram(
                np_histogram=explore_return_hist
            )
            log_dict["explore/explore_length_hist"] = wandb.Histogram(
                np_histogram=explore_length_hist
            )

            # reset record of return and length, to record the new trajectory's return and length distribution
            buffer.p_return_list.clear()
            buffer.p_length_list.clear()

            # log_dict.update(val_dict)
            wandb_logger.log(log_dict, step=step)

        log_dict["time/iteration_step_time"] = time.time() - B

        if random.randint(0, cfg.log_every) == 0:
            logger.info(f"Step {step}")
            wandb_logger.log(log_dict, step=step)

        step += 1
        if step >= cfg.num_train_steps or buffer.total_step > cfg.explore_steps:
            break

    torch.save(
        {
            "model": learner.mtm.state_dict(),
            "optimizer": learner.mtm_optimizer.state_dict(),
            "step": step,
        },
        f"mtm_{step}.pt",
    )
    torch.save(learner.iql.state_dict(), f"iql_{step}.pt")


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()
