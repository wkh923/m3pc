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

from research.mtm.masks import (
    MaskType,
    create_bc_mask,
    create_forward_dynamics_mask,
    create_full_random_masks,
    create_goal_n_reaching_masks,
    create_goal_reaching_masks,
    create_inverse_dynamics_mask,
    create_random_autoregressize_mask,
    create_random_bc_masks,
    create_random_mask,
    create_random_masks,
    create_rcbc_mask,
    maybe_add_rew_to_mask,
)
from research.utils.plot_utils import PlotHandler as ph
from research.jaxrl.utils import make_env
from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.distributed_utils import DistributedParams, get_distributed_params
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
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


@torch.no_grad()
def make_plots_with_masks(
    predict_fn,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    masks_list: List[Dict[str, torch.Tensor]],
    RTG_ratio: List[float],
    prefixs: List[str],
    batch_size: int = 1,
    max_n_plots: int = 3,
):
    eval_logs = {}
    for masks, prefix in zip(masks_list, prefixs):
        eval_name = f"{prefix}_eval"

        encoded_trajectories = tokenizer_manager.encode(trajectories)
        decoded_gt_trajectories = tokenizer_manager.decode(encoded_trajectories)
        # print("mask in device", masks["states"].device)
        # print("trajectories in device", trajectories["states"].device)
        predictions = predict_fn(encoded_trajectories, masks)
        decoded_trajs = tokenizer_manager.decode(predictions)

        pertubed_trajectories_list = []

        for RGT_scale in RTG_ratio:
            pertubed_trajectories = trajectories.copy()
            pertubed_trajectories["returns"] = trajectories["returns"] * RGT_scale
            encoded_pertubed_trajectories = tokenizer_manager.encode(
                pertubed_trajectories
            )
            pertubed_predictions = predict_fn(encoded_pertubed_trajectories, masks)
            decoded_pertubed_trajs = tokenizer_manager.decode(pertubed_predictions)
            pertubed_trajectories_list.append(decoded_pertubed_trajs)

        print(" ==== batch size ==== :", batch_size)

        for k, _ in decoded_trajs.items():
            # traj = trajectories[k][batch_idx].cpu().numpy()
            # if len(traj.shape) == 1:
            #     traj = traj[:, None]
            # pred_traj = decoded_trajs[k][batch_idx].cpu().numpy()
            # if len(pred_traj.shape) == 1:
            #     pred_traj = pred_traj[:, None]

            i_mse = (
                F.mse_loss(trajectories[k], decoded_trajs[k], reduction="none")
                .mean(dim=(-1))
                .cpu()
                .numpy()
            )
            # print("i_mse size", i_mse.shape)
            if k + "/" + eval_name not in eval_logs:
                eval_logs[k + "/" + eval_name] = np.zeros((batch_size, 8))
                eval_logs[k + "/" + eval_name] = i_mse
            for pertube_scale, pertubed_trajs in zip(
                RTG_ratio, pertubed_trajectories_list
            ):
                p_mse = (
                    F.mse_loss(decoded_trajs[k], pertubed_trajs[k], reduction="none")
                    .mean(dim=(-1))
                    .cpu()
                    .numpy()
                )
                log_name = k + "/" + eval_name + f"noise_{pertube_scale}"
                if log_name not in eval_logs:
                    eval_logs[log_name] = np.zeros((batch_size, 8))
                    eval_logs[log_name] = p_mse

    for k, v in eval_logs.items():
        with ph.plot_context() as (fig, ax):
            # Transpose data to match the expected shape for boxplot
            # Each row now represents a dataset for one box
            ax.boxplot(v, showfliers=False)
            ax.set_xlabel("Traj Index")
            ax.set_ylabel("SME Loss")
            ax.set_title(k)

            # Create a custom name for this plot in your wandb logs
            plot_name = k

            # Log the plot to wandb
            wandb.log({plot_name: wandb.Image(ph.plot_as_image(fig))})
        # for pertube_scale in RTG_ratio:
        #     log_name = k + f"noise_{pertube_scale}"
        #     with ph.plot_context() as (fig, ax):
        #         # Transpose data to match the expected shape for boxplot
        #         # Each row now represents a dataset for one box
        #         ax.boxplot(eval_logs[log_name])
        #         ax.set_xlabel("Traj Index")
        #         ax.set_ylabel("SME Loss")
        #         ax.set_title(log_name)

        #         # Create a custom name for this plot in your wandb logs
        #         plot_name = log_name

        #         # Log the plot to wandb
        #         wandb.log({plot_name: wandb.Image(ph.plot_as_image(fig))})

    return


def create_eval_logs_states_actions(
    predict_fn: Callable,
    trajectories: Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    rewards: bool = False,
    batch: int = 1,
) -> Dict[str, Any]:
    eval_logs = {}
    assert "states" in trajectories
    assert "actions" in trajectories
    device = trajectories["states"].device
    seq_len = trajectories["states"].shape[1]

    obs_use_mask_list = []
    actions_use_mask_list = []

    # make mask from the first state to the last state
    for masked_len in range(seq_len):
        obs_mask = np.ones(seq_len)
        obs_mask[masked_len + 1 :] = 0
        actions_mask = np.ones(seq_len)
        actions_mask[masked_len:] = 0
        obs_use_mask_list.append(obs_mask)
        actions_use_mask_list.append(actions_mask)

    masks_list = []
    for obs_mask, actions_mask in zip(obs_use_mask_list, actions_use_mask_list):
        masks_list.append(
            {
                "states": torch.from_numpy(obs_mask).to(device),
                "actions": torch.from_numpy(actions_mask).to(device),
            }
        )
        if rewards:
            masks_list[-1]["rewards"] = masks_list[-1]["states"].clone()
            masks_list[-1]["returns"] = masks_list[-1]["states"].clone()

    prefixs = [
        "fd_in1_out8",
        "fd_in2_out7",
        "fd_in3_out6",
        "fd_in4_out5",
        "fd_in5_out4",
        "fd_in6_out3",
        "fd_in7_out2",
        "fd_in8_out1",
    ]
    RTG_ratio = [0.2, 0.4, 0.7, 2.0, 4.0]
    return make_plots_with_masks(
        predict_fn,
        trajectories,
        tokenizer_manager,
        masks_list,
        RTG_ratio,
        prefixs,
        max_n_plots=1,
        batch_size=batch,
    )


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

    beam_width: int = 256
    """The number of candidates retained during beam search"""

    trans_buffer_update: bool = True  # [True, False]
    trans_buffer_init_method: str = "top_trans"  # ["top_trans", "top_traj", "random"]

    critic_update: bool = True  # [True, False]


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
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    wandb_cfg_log = WandBLoggerConfig(
        experiment_id=hydra_cfg.wandb.experiment_name + "_" + current_time,
        # experiment_id=f"{dp.job_id}-{dp.rank}",
        project=hydra_cfg.wandb.project,
        entity=hydra_cfg.wandb.entity or None,
        resume=hydra_cfg.wandb.resume,
        group=dp.job_id,
    )
    wandb_logger = WandBLogger(wandb_cfg_log, wandb_cfg_log_dict)
    logger.info(f"eval offline model with gradually masked traj")
    model = learner.mtm
    model.eval()
    for _ in range(1):
        val_batch = next(iter(val_loader))
        val_batch = {k: v.to(cfg.device) for k, v in val_batch.items()}

        create_eval_logs_states_actions(
            model, val_batch, tokenizer_manager, rewards=True, batch=cfg.traj_batch_size
        )


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()
