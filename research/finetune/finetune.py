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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import torch.dis
import tqdm


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

import gym


def create_rcbc_mask(traj_length: int, position: int, 
                device: str) -> Dict[str, torch.tensor]:
    """Return based policy: output an anction at position 
    based on previous knowledge"""
    
    assert position < traj_length
    state_mask = np.zeros(traj_length)
    state_mask[:position] = 1
    return_mask = np.zeros(traj_length)
    return_mask[:position] = 1
    
    action_mask = np.ones(traj_length)
    action_mask[position:] = 0
    reward_mask = np.ones(traj_length)
    reward_mask[position:] = 0
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

def create_forward_dynamics_mask(traj_length: int, position: int,
                            device: str) -> Dict[str, torch.tensor]:
    
    """Forward dynamics prediction"""
    
    assert position < traj_length and position > 0
    state_mask = np.zeros(traj_length)
    state_mask[:position] = 1
    return_mask = np.zeros(traj_length)
    return_mask[:position] = 1
    action_mask = np.zeros(traj_length)
    action_mask[:position] = 1
    reward_mask = np.zeros(traj_length)
    reward_mask[:position] = 1
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

def create_reward_estimation_mask(traj_length: int, position: int,
                            device: str) -> Dict[str, torch.tensor]:
    """Reward estimation"""
    assert position < traj_length
    state_mask = np.zeros(traj_length)
    state_mask[:position] = 1
    return_mask = np.zeros(traj_length)
    return_mask[:position] = 1
    action_mask = np.zeros(traj_length)
    action_mask[:position] = 1
    
    reward_mask = np.ones(traj_length)
    reward_mask[position:] = 0
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

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

import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        action_shape,
        observation_shape,
        device,
        max_buffer_size: int = 10000,
        traj_length: int = 8
    ):
        self.device = device
        self.action_shape = action_shape
        self.observation_shape = observation_shape
        self.max_buffer_size = max_buffer_size
        self.traj_length = traj_length
        
        # Initialize buffer with empty trajectories
        self.observations = np.zeros((0, self.traj_length) + self.observation_shape)
        self.actions = np.zeros((0, self.traj_length) + self.action_shape)
        self.rewards = np.zeros((0, self.traj_length, 1))
        self.returns = np.zeros((0, self.traj_length, 1))

    def create_zero_trajectory(self):   
        """Creates an empty trajectory with no samples."""
        zero_trajectory = {
            "states": np.zeros((self.traj_length,) + self.observation_shape),
            "actions": np.zeros((self.traj_length,) + self.action_shape),
            "rewards": np.zeros((self.traj_length, 1)),
            "returns": np.zeros((self.traj_length, 1))
        }
        torch_zero_trajectories = {
            k: torch.tensor(v, device=self.device)[None] for k, v in zero_trajectory.items()}
        return torch_zero_trajectories
    
    def add_traj_to_buffer(self, trajectory):
        """Add a new tensor trajectory to the buffer and ensure its size never exceeds max_buffer_size."""
        # Convert tensors to numpy
        trajectory_np = {k: v.cpu().numpy() for k, v in trajectory.items()}
        
        assert trajectory_np["actions"].shape[1] == self.traj_length, "Trajectory length does not match buffer's traj_length."
        
        N, _, _ = trajectory_np["actions"].shape
        overflow = max(0, self.observations.shape[0] + N - self.max_buffer_size)
        
        if overflow > 0:
            # Remove the oldest trajectories to fit the new ones
            self.observations = self.observations[overflow:]
            self.actions = self.actions[overflow:]
            self.rewards = self.rewards[overflow:]
            self.returns = self.returns[overflow:]

        # Append new trajectory
        self.observations = np.concatenate((self.observations, trajectory_np["states"]), axis=0)
        self.actions = np.concatenate((self.actions, trajectory_np["actions"]), axis=0)
        self.rewards = np.concatenate((self.rewards, trajectory_np["rewards"]), axis=0)
        self.returns = np.concatenate((self.returns, trajectory_np["returns"]), axis=0)

    def rollout_trajectories(self, env, model, num_trajectories, tokenizer_manager, percentage=1.0,
                             clip_min=-1.0, clip_max=1.0, action_noise_std=0.2):
        """Use RCBC inference to collect trajectories"""
        
        model.eval()
        return_max = tokenizer_manager.tokenizers["returns"].stats.max
        return_min = tokenizer_manager.tokenizers["returns"].stats.min
        
        return_value = return_min + (return_max - return_min) * percentage
        return_to_go = float(return_value)
        returns = return_to_go * np.ones((self.traj_length, 1))
        
        
        for _ in range(num_trajectories):
            torch_trajectory = self.create_zero_trajectory()
            torch_trajectory["returns"] = torch.tensor(returns, device=self.device).unsqueeze(0)
            state = env.reset()
            for pos in range(self.traj_length):
                torch_trajectory["states"][0, pos] = torch.tensor(state, device=self.device)
                torch_rcbc_mask = create_rcbc_mask(self.traj_length, pos, self.device)
                encoded_trajectory = tokenizer_manager.encode(torch_trajectory)
                with torch.no_grad():
                    prediction = model(encoded_trajectory, torch_rcbc_mask)
                decode = tokenizer_manager.decode(prediction)
                
                action = decode["actions"][0, pos].cpu().numpy()
                action += np.random.normal(0, action_noise_std, size=action.shape)
                if clip_min is not None and clip_max is not None:
                    action = np.clip(action, clip_min, clip_max)
                
                next_state, reward, done, _ = env.step(action)
                
                torch_trajectory["actions"][0, pos] = torch.tensor(action, device=self.device)
                torch_trajectory["rewards"][0, pos] = torch.tensor(reward, device=self.device)
                
                if done:
                    break
                state = next_state
                
            self.add_traj_to_buffer(torch_trajectory)
            
        model.train()


def compute_mtm_loss(
    
    cfg: RunConfig,
    model:MTM,
    critic_model,
    batch:Dict[str, torch.Tensor],
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, torch.Tensor],

):
    
    
    def compute_target_cem_action(trajectory, traj_length, pos, device, policy_std, discount,
                            n_iter=5, n_rsamples=900, n_policy_samples=100, top_k=100):
        
        
        assert pos < traj_length - 1
        action_dim = trajectory["actions"].shape[-1]
        cem_init_mean = tokenizer_manager.tokenizers["actions"].stats.mean
        cem_init_std = tokenizer_manager.tokenizers["actions"].stats.std
        cem_init_mean = torch.tensor(cem_init_mean, device=device).unsqueeze(0)
        cem_init_std = torch.tensor(cem_init_std, device=device).unsqueeze(0)
        with torch.no_grad():
            for it in range(n_iter):
                
                
                # Generate action samples
                action_rsamples = cem_init_mean + cem_init_std * torch.randn(n_rsamples, action_dim, device=device)
                action_policy = trajectory["actions"][:, pos, :]
                action_policy_samples = action_policy + policy_std * torch.randn(n_policy_samples, action_dim, device=device)
                
                
                # Concatenate random and policy action samples
                combined_actions = torch.cat([action_rsamples, action_policy_samples], dim=0)
                
                # Use the model to predict next states and rewards for each action sample
                batch = {k: v.repeat(combined_actions.shape[0], 1, 1) for k, v in trajectory.items()}
                batch["actions"][:, pos, :] = combined_actions
                encoded_batch = tokenizer_manager.encode(batch)
                torch_reward_mask = create_reward_estimation_mask(traj_length=traj_length, position=pos, device=device)
                pred = model(encoded_batch, torch_reward_mask)
                rewards = tokenizer_manager.decode(pred)["rewards"][:, pos, :]
                
                batch["rewards"][:, pos, :] = rewards
                encoded_batch = tokenizer_manager.encode(batch)
                torch_fd_mask = create_forward_dynamics_mask(traj_length=traj_length, position=pos+1, device=device)
                pred = model(encoded_batch, torch_fd_mask)
                next_states = tokenizer_manager.decode(pred)["states"][:, pos+1, :]
                
                total_rewards = rewards
                
                batch["states"][:, pos+1, :] = next_states
                encoded_batch = tokenizer_manager.encode(batch)
                torch_rcbc_mask = create_rcbc_mask(traj_length=traj_length, position=pos+1, device=device)
                pred = model(encoded_batch, torch_rcbc_mask)
                decode = tokenizer_manager.decode(pred)
                last_actions = decode["actions"][:, -1, :]
                
                if pos < traj_length - 2:
                    future_rewards = decode[:, pos+1:-1, ]
                    next_states = decode[:, -1, :]
                    for i in range(future_rewards.shape[1]):
                        total_rewards += future_rewards[:, i] * (discount ** (i + 1))
                
                total_rewards = total_rewards.squeeze(-1)
                last_states = next_states
                total_rewards += critic_model(last_states, last_actions) * (discount ** (traj_length - pos))
                
                _, sorted_indices = torch.sort(total_rewards, descending=True)
                top_k_actions = combined_actions[sorted_indices[0:top_k]]
                
                cem_init_mean = torch.mean(top_k_actions, dim=0)
                cem_init_std = torch.std(top_k_actions, dim=0, unbiased=False)
                
        return cem_init_mean, cem_init_std

    
    loss_dict = {}
    encoded_batch = tokenizer_manager.encode(batch)
    batch_size = encoded_batch.shape[0]
    
    dynamic_loss = 0
    
    
    for pos in range(cfg.traj_length):
        torch_fd_mask = create_forward_dynamics_mask(
            traj_length=cfg.traj_length, position=pos+1, device=cfg.device)
        pred = model(encoded_batch, torch_fd_mask)["states"][:, pos+1, ...]
        target = encoded_batch["states"][:, pos+1, ...]
        
        if discrete_map["states"]:
            pos_dynamic_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
                ).unsqueeze(3)
        else:
            # apply normalization
                if cfg.norm == "l2":
                    target = target / torch.norm(target, dim=-1, keepdim=True)
                elif cfg.norm == "mae":
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / (var + 1.0e-6) ** 0.5

                pos_dynamic_loss = nn.MSELoss(pred, target)

            # pos_loss shape = [batch_size, 1, P, 1]
        
        dynamic_loss += pos_dynamic_loss.mean()
    dynamic_loss /= cfg.traj_length
    loss_dict["fd_loss"] = dynamic_loss
    
    reward_loss = 0
    
    for pos in range(cfg.traj_length):
        torch_reward_mask = create_reward_estimation_mask(cfg.traj_length, pos, cfg.device)
        pred = model(encoded_batch, torch_reward_mask)["rewards"][:, pos, ...]
        target = encoded_batch["rewards"][:, pos, ...]
        
        if cfg.norm == "l2":
            target = target / torch.norm(target, dim=-1, keepdim=True)
        elif cfg.norm == "mae":
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        pos_reward_loss = nn.MSELoss(pred, target)

        # pos_loss shape = [batch_size, 1, P, 1]
        
        reward_loss += pos_reward_loss.mean()
    reward_loss /= cfg.traj_length
    
    loss_dict["r_loss"] = reward_loss
    
    policy_loss = 0
    for pos in range(cfg.traj_length):
        torch_rcbc_mask = create_rcbc_mask(cfg.traj_length, pos, cfg.device)
        pred = model(encoded_batch, torch_rcbc_mask)
        actions = tokenizer_manager.decode(pred)["actions"][:, pos, :]
        for traj_indx in range(batch_size):
            trajectory = {k:v[traj_indx].unsqueeze(0) for k, v in batch.items()}
            target_mean, target_std = compute_target_cem_action(trajectory, cfg.traj_length, pos,
                cfg.device, cfg.policy_std, cfg.discount, cfg.n_iter, cfg.n_rsamples, cfg.n_policy_samples, cfg.top_k)
            policy_loss += ((actions[traj_indx] - target_mean.squeeze(0)) ** 2) / (target_std.squeeze(0) ** 2)
    
    policy_loss /= (batch_size * cfg.traj_length)
    
    loss_dict["a_loss"] = policy_loss
        
    return loss_dict
    
def compute_q_loss(
    
    cfg: RunConfig,
    model:MTM,
    critic_model,
    critic_target_model,
    tokenizer_manager: TokenizerManager,
    batch: Dict[str, torch.Tensor],
    
):  
    model.eval()
    #convert batch data structure
    states = batch["states"][:, :-1]
    actions = batch["actions"][:, :-1]
    rewards = batch["rewards"][:, :-1]
    next_states = batch["states"][:, 1:]
    next_actions = torch.zeros_like(actions)
    
    # Flatten the data for compatibility with the model
    states = states.reshape(-1, states.shape[-1])
    actions = actions.reshape(-1, actions.shape[-1])
    rewards = rewards.reshape(-1)
    next_states = next_states.reshape(-1, next_states.shape[-1])
    
    # Predicted Q-values for the current state-action pairs
    predicted_q_values = critic_model(states, actions)
    
    
    with torch.no_grad():
        for pos in range(actions.shape[1]):
            torch_rcbc_mask = create_rcbc_mask(traj_length=actions.shape[1]+1, position=pos+1, device=cfg.device)
            encode = tokenizer_manager.encode(batch)
            pred = model(encode, torch_rcbc_mask)
            next_actions[:, pos] = tokenizer_manager.decode(pred)["actions"][:, pos+1]
        next_actions = next_actions.reshape(-1, next_actions.shape[-1])
        
        target_next_q_values = critic_target_model(next_states, next_actions)
    target_q_values = rewards + (cfg.discount * target_next_q_values)
    
    q_loss = nn.MSELoss(predicted_q_values, target_q_values)
    
    model.train()
    
    return q_loss

class Learner(object):
    def __init__(self,
                 cfg: RunConfig,
                 model: MTM,
                 critic_model,
                 tokenizer_manager: TokenizerManager,
                 discrete_map: Dict[str, torch.Tensor]
                 ):
        self.cfg = cfg
        self.mtm = model
        self.critic_model = critic_model
        self.critic_target = critic_model
        self.tokenizer_manager = tokenizer_manager
        self.discrete_map = discrete_map
        self.mtm_optimizer = MTM.configure_optimizers(
            self.mtm,
            learning_rate = self.cfg.learning_rate,
            weght_decay = self.cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        def _schedule(step):
            # warmp for 1000 steps
            if step < self.cfg.warmup_steps:
                return step / cfg.warmup_steps

            # then cosine decay
            assert self.cfg.num_train_steps > self.cfg.warmup_steps
            step = step - self.cfg.warmup_steps
            return 0.5 * (
                1 + np.cos(step / (self.cfg.num_train_steps - self.cfg.warmup_steps) * np.pi)
            )

        self.mtm_scheduler = LambdaLR(self.mtm_optimizer, lr_lambda=_schedule)
        
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(),
                                                 lr=self.cfg.learning_rate,
                                                 weght_decay = self.cfg.weight_decay,
                                                 betas=(0.9, 0.999))
        self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda=_schedule)
        
        self.mtm.train()
        self.critic_model.train()
        self.critic_target.eval()
        
    def mtm_update(self,
                   batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   scheduler: Callable):
        
        #compute the loss
        loss_dict = compute_mtm_loss(self.cfg, self.mtm, self.critic_model,
                                     batch, self.tokenizer_manager, self.discrete_map)
        loss_weight = self.cfg.loss_weight
        
        loss = 0
        log_dict = {}
        for loss_key, weight in loss_weight.items():
            loss_dict[loss_key] *= weight
            loss += loss_dict[loss_key]
            log_dict[f"train/{loss_key}"] = loss_dict[loss_key].item()
        log_dict[f"train/loss"] = loss.item()
        log_dict["train/lr"] = scheduler.get_last_lr()[0]
        
        #backprop
        self.mtm.zero_grad(set_to_none=True)
        loss.backward()
        self.mtm_optimizer.step()
        self.mtm_scheduler.step()
        
        return log_dict
    
    def critic_update(self,
                      batch: Dict[str, torch.Tensor]):
        critic_loss = compute_q_loss(self.cfg, self.mtm, self.critic_model, self.critic_target,
                                     self.tokenizer_manager, batch)
        
        log_dict = {}
        log_dict[f"train/q_loss"] = critic_loss.item()
        
        #backprop
        self.critic_model.zero_grad(set_to_none=True)
        critic_loss.backward()
        
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        
    def critic_target_soft_update(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
    
    
    def action_sample(self, observation, action_dim, percentage=1.0):
        zero_trajectory = {
            "states": np.zeros((self.cfg.traj_length,) + observation.shape[-1]),
            "actions": np.zeros((self.cfg.traj_length,) + action_dim),
            "rewards": np.zeros((self.cfg.traj_length, 1)),
            "returns": np.zeros((self.cfg.traj_length, 1))
        }
        torch_zero_trajectories = {
            k: torch.tensor(v, device=self.cfg.device)[None] for k, v in zero_trajectory.items()}
        torch_zero_trajectories["states"][0, 0] = torch.tensor(observation)
        return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
        return_min = self.tokenizer_manager.tokenizers["returns"].stats.min
        
        return_value = return_min + (return_max - return_min) * percentage
        return_to_go = float(return_value)
        returns = return_to_go * np.ones((self.traj_length, 1))
        torch_zero_trajectories["returns"][0,...] = torch.tensor(returns)
        
        torch_rcbc_mask = create_rcbc_mask(self.cfg.traj_length, 0, self.cfg.device)
        encode = self.tokenizer_manager.encode(torch_zero_trajectories)
        with torch.no_grad():
            pred = self.mtm(encode, torch_rcbc_mask)
        action = self.tokenizer_manager.decode(pred)["actions"][0, 0, :].cpu().np()
        
        return action
    
    
    
    def evaluate(self,
                 env: gym.Env,
                 num_episodes: int,
                 disable_tqdm: bool = True,
                 verbose: bool = False,
                 all_results: bool = False,
                 num_videos: int = 3,
                ) -> Dict[str, Any]:
        
        
        self.mtm.eval()
        stats: Dict[str, Any] = defaultdict(list)
        successes = None

        pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)

        videos = []
        
        for i in pbar:
            observation, done = env.reset(), False
            trajectory_history = Trajectory.create_empty(env.observation_space.shape, env.action_space.shape)
            if len(videos) < num_videos:
                try:
                    imgs = [env.sim.render(64, 48, camera_name="track")[::-1]]
                except:
                    imgs = [env.render()[::-1]]
            
            while not done:
                action = self.action_sample(observation, env.action_space.shape, percentage=1.0)
                action = np.clip(action, self.cfg.clip_min, self.cfg.clip_max)
                new_observation, reward, done, info = env.step(action)
                trajectory_history = trajectory_history.append(observation, action, reward)
                observation = new_observation
                if len(videos) < num_videos:
                    try:
                        imgs.append(env.sim.render(64, 48, camera_name="track")[::-1])
                    except:
                        imgs.append(env.render()[::-1])
            
            if len(videos) < num_videos:
                videos.append(np.array(imgs[:-1]))
                
            if "episode" in info:
                for k in info["episode"].keys():
                    stats[k].append(float(info["episode"][k]))
                    if verbose:
                        print(f"{k}:{info['episode'][k]}")
                    
                ret = info["episode"]["return"]
                mean = np.mean(stats["return"])
                pbar.set_description(f"iter={i}\t last={ret:.2f} mean={mean}")
                if "is_success" in  info:
                    if successes is None:
                        successes = 0.0
                    successes += info["is_success"]
            
            else:
                stats["return"].append(trajectory_history.rewards.sum())
                stats["length"].append(len(trajectory_history.rewards))
                stats["achieved"].append(int(info["goal_achieved"]))
        
        new_stats = {}
        for k, v in stats.items():
            new_stats[k + "_mean"] = float(np.mean(v))
            new_stats[k + "_std"] = float(np.std(v))
        if all_results:
            new_stats.update(stats)
        stats = new_stats

        if successes is not None:
            stats["success"] = successes / num_episodes

        return stats, videos
                
        

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
    
    buffer = ReplayBuffer(action_shape, observation_shape, device=cfg.device,
                          max_buffer_size=cfg.buffer_size,
                          traj_length=cfg.traj_length)
    zero_traj = buffer.create_zero_trajectory()
    
    
    data_shapes = {}
    for k, v in tokenizer_manager.encode(zero_traj).items():
        data_shapes[k] = v.shape[-2:]
    
    print(f"Data shapes: {data_shapes}")
    logger.info(f"Data shapes: {data_shapes}")
    
    pretrain_model = model_config.create(data_shapes, cfg.traj_length)
    pretrain_model.load_state_dict(torch.load(pretrain_model_path + "model_60000.pt")["model"])
    pretrain_model.to(cfg.device)
    
    learner = Learner(cfg, pretrain_model, critic_model, tokenizer_manager, discrete_map)
    
    
    buffer.rollout_trajectories(env=env, model=pretrain_model, num_trajectories=cfg.buffer_size, 
                            tokenizer_manager=tokenizer_manager, percentage=1.0, 
                            clip_max=cfg.clip_min, clip_min=cfg.clip_max, action_noise_std=cfg.action_noise_std)
    

@hydra.main(config_path=".", config_name="config", version_base = "1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    # logger.info(hydra_data)
    main(hydra_data)
  


if __name__ == "__main__":
    configure_jobs()