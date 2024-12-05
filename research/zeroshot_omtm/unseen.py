import os
import pprint
from dataclasses import dataclass
from typing import Dict, Tuple

import hydra
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.parallel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader

from research.finetune_omtm.replay_buffer import ReplayBuffer
from research.jaxrl.utils import make_unseen_env
from research.logger import logger
from research.omtm.datasets.base import DatasetProtocol
from research.omtm.distributed_utils import DistributedParams, get_distributed_params
from research.omtm.tokenizers.base import Tokenizer, TokenizerManager
from research.omtm.utils import set_seed_everywhere
from research.zeroshot_omtm.learner import Learner


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

    index_jump: int = 0

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


def main(hydra_cfg):
    dp: DistributedParams = get_distributed_params()
    torch.cuda.set_device(hydra_cfg.local_cuda_rank)

    hydra_cfg.goal_mask in ["id", "piid"]
    two_stage = True if hydra_cfg.goal_mask == "piid" else False
    list_stage = True if hydra_cfg.goal_mask == "piid_allout" else False

    # TODO: get goal_mask from hydra_cfg

    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.finetune_args)
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    cfg.traj_length = hydra_cfg.pretrain_args.traj_length
    cfg.env_name = hydra_cfg.pretrain_args.env_name

    set_seed_everywhere(cfg.seed)
    pprint.pp(cfg)

    current_path = os.getcwd()
    print(current_path)
    logger.info(f"Working directory: {current_path}")
    with open("unseen_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))

    pretrain_model_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.pretrain_model_path)
    )

    waypoint_path = os.path.normpath(
        os.path.join(current_path, hydra_cfg.waypoints_path)
    )
    train_dataset: DatasetProtocol
    val_dataset: DatasetProtocol

    train_dataset, val_dataset, train_raw_dataset = hydra.utils.call(
        hydra_cfg.pretrain_dataset, seq_steps=cfg.traj_length
    )
    env = make_unseen_env(cfg.env_name, cfg.seed)
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

    learner.mtm.eval()

    val_dict, _ = learner.shot(
        num_episodes=10,
        episode_rtg_ref=buffer.values_up_bound,
        way_points_path=waypoint_path,
        two_stage=two_stage,
        list_stage=list_stage,
    )


@hydra.main(config_path=".", config_name="config_cheeta", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)


if __name__ == "__main__":
    configure_jobs()
