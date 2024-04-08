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

    use_masked_loss: bool = True

    loss_weight: Dict[str, float] = field(
        default_factory=lambda: {
            "actions": 1.0,
            "states": 1.0,
            "returns": 1.0,
            "rewards": 1.0,
        }
    )

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
    exploration_noise_std: float = 0.1
    critic_noise_std: float = 0.1

    beam_width: int = 256
    """The number of candidates retained during beam search"""

    horizon: int = 4
    """The horizon for planning, horizon=1 means critic guided search"""
    rtg_percent: float = 1.0

    trans_buffer_update: bool = True  # [True, False]
    trans_buffer_init_method: str = "top_trans"  # ["top_trans", "top_traj", "random"]
    filter_short_traj: bool = False  # [True, False]

    critic_update: bool = True  # [True, False]
    critic_scratch: bool = False  # [True, False]
    mtm_update: bool = True  # [True, False]


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
    pretrain_critic1_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.pretrain_critic1_path)
    )
    pretrain_critic2_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.pretrain_critic2_path)
    )
    pretrain_value_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.pretrain_value_path)
    )
    pretrain_actor_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.pretrain_actor_path)
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
        pretrain_critic1_path,
        pretrain_critic2_path,
        pretrain_value_path,
        pretrain_actor_path,
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

    step = 0
    
    logger.info(f"starting from step={step}")

    episode = 0
    while True:
        B = time.time()
        log_dict = {}
        log_dict["train/episodes"] = episode

        start_time = time.time()

        if cfg.critic_update == True:

            for critic_iter in range(cfg.v_iter_per_mtm):
                experiences = buffer.trans_sample()
                value_log = learner.value_update(experiences)
                critic_log = learner.critic_update(experiences)
                policy_log = learner.policy_update(experiences)
                learner.critic_target_soft_update()

            log_dict.update(critic_log)
            log_dict.update(value_log)
            log_dict.update(policy_log)

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

        if cfg.mtm_update == True:
            mtm_log = learner.mtm_update(batch, data_shapes, discrete_map)
            log_dict.update(mtm_log)

        log_dict["time/train_step"] = time.time() - start_time

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
            torch.save(
                {
                    "model": learner.critic1.state_dict(),
                    "optimizer": learner.critic1_optimizer.state_dict(),
                    "step": step,
                },
                f"critic1_{step}.pt",
            )
            torch.save(
                {
                    "model": learner.critic2.state_dict(),
                    "optimizer": learner.critic2_optimizer.state_dict(),
                    "step": step,
                },
                f"critic2_{step}.pt",
            )
            torch.save(
                {
                    "model": learner.value.state_dict(),
                    "optimizer": learner.value_optimizer.state_dict(),
                    "step": step,
                },
                f"value_{step}.pt",
            )

            try:
                if step > 3 * cfg.save_every:
                    remove_step = step - 3 * cfg.save_every
                    if (remove_step // cfg.save_every) % 10 != 0:
                        os.remove(f"mtm_{remove_step}.pt")
                        os.remove(f"critic1_{remove_step}.pt")
                        os.remove(f"critic2_{remove_step}.pt")
                        os.remove(f"value_{remove_step}.pt")
            except Exception as e:
                logger.error(f"Failed to remove model file! {e}")

        if step % cfg.eval_every == 0:
            start_time = time.time()
            learner.mtm.eval()
            learner.critic1.eval()
            learner.critic2.eval()
            learner.value.eval()

            val_batch = next(iter(val_loader))
            val_batch = {
                k: v.to(cfg.device, non_blocking=True) for k, v in val_batch.items()
            }
            (loss, losses, masked_losses, masked_c_losses, entropy) = learner.compute_mtm_loss(
                val_batch, data_shapes, discrete_map, learner.mtm.temperature().detach()
            )

            log_dict["eval/loss"] = loss.item()
            for k, v in losses.items():
                log_dict[f"eval/loss_{k}"] = v
            for k, v in masked_losses.items():
                log_dict[f"eval/masked_loss_{k}"] = v
            for k, v in masked_c_losses.items():
                log_dict[f"eval/masked_c_loss_{k}"] = v
            log_dict[f"eval/entropy"] = entropy.item()

            val_dict = learner.evaluate(num_episodes=10)
            val_dict.update(learner.evaluate_policy(num_episodes=10))

            learner.mtm.train()
            learner.critic1.train()
            learner.critic2.train()
            learner.value.train()
            val_dict["time/eval_step_time"] = time.time() - start_time

            explore_return_hist = np.histogram(
                buffer.p_return_list, bins=list(range(0, 3501, 50))
            )
            explore_length_hist = np.histogram(
                buffer.p_length_list, bins=list(range(0, 1001, 20))
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

            # plt.figure()
            # plt.hist(
            #     buffer.p_return_list,
            #     bins=list(range(0, 3501, 500)),
            #     color="blue",
            #     edgecolor="black",
            # )
            # plt.title("Histogram of Return")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # log_dict["eval/new_trajectory_return"] = wandb.Image(plt)

            # plt.figure()
            # plt.hist(
            #     buffer.p_length_list,
            #     bins=list(range(0, 1001, 200)),
            #     color="blue",
            #     edgecolor="black",
            # )
            # plt.title("Histogram of Length")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # log_dict["eval/new_trajectory_length"] = wandb.Image(plt)
            log_dict.update(val_dict)
            wandb_logger.log(log_dict, step=step)

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
        f"mtm_{step}.pt",
    )
    torch.save(
        {
            "model": learner.critic1.state_dict(),
            "optimizer": learner.critic1_optimizer.state_dict(),
            "step": step,
        },
        f"critic1_{step}.pt",
    )
    torch.save(
        {
            "model": learner.critic2.state_dict(),
            "optimizer": learner.critic2_optimizer.state_dict(),
            "step": step,
        },
        f"critic2_{step}.pt",
    )
    torch.save(
        {
            "model": learner.value.state_dict(),
            "optimizer": learner.value_optimizer.state_dict(),
            "step": step,
        },
        f"value_{step}.pt",
    )


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()
