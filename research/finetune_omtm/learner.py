from research.finetune_omtm.masks import *
from research.finetune_omtm.model import *
from research.omtm.models.mtm_model import omtm
from research.omtm.tokenizers.base import TokenizerManager
from research.omtm.datasets.sequence_dataset import Trajectory
from research.jaxrl.utils import make_env
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import tqdm
import wandb
import gym
import os


class Learner(object):
    def __init__(
        self,
        cfg,
        env: gym.Env,
        data_shapes,
        model_config,
        pretrain_model_path,
        obs_mean,
        obs_std,
        tokenizer_manager: TokenizerManager,
        discrete_map: Dict[str, torch.Tensor],
    ):
        self.cfg = cfg
        self.env = env
        self.mtm = model_config.create(data_shapes, cfg.traj_length, discrete_map)
        self.mtm.load_state_dict(
            torch.load(pretrain_model_path, map_location="cpu")["model"]
        )
        self.mtm.to(cfg.device)

        self.tokenizer_manager = tokenizer_manager
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.discrete_map = discrete_map
        self.mtm_optimizer = omtm.configure_optimizers(
            self.mtm,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        self.eps = 1e-5

        def _schedule(step):
            return 0.5 * (1 + np.cos(step / cfg.num_train_steps * np.pi))

        self.mtm_scheduler = LambdaLR(self.mtm_optimizer, lr_lambda=_schedule)
        self.temp_optimizer = torch.optim.Adam(
            [self.mtm.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        q_network = TwinQ(state_dim, action_dim, self.obs_mean, self.obs_std).to(
            self.cfg.device
        )
        v_network = ValueFunction(state_dim, self.obs_mean, self.obs_std).to(
            self.cfg.device
        )
        actor = (
            GaussianPolicy(
                state_dim,
                action_dim,
                float(self.env.action_space.high[0]),
                obs_mean=self.obs_mean,
                obs_std=self.obs_std,
                dropout=0.0,
            )
        ).to(self.cfg.device)

        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=self.cfg.v_lr)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=self.cfg.critic_lr)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.v_lr)
        iql_kwargs = {
            "max_action": float(self.env.action_space.high[0]),
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": self.cfg.discount,
            "tau": self.cfg.tau,
            "device": self.cfg.device,
            # IQL
            "beta": 3.0,
            "iql_tau": self.cfg.expectile,
            "max_steps": self.cfg.num_train_steps * self.cfg.v_iter_per_mtm
            + self.cfg.warmup_steps,
        }
        self.iql = ImplicitQLearning(**iql_kwargs)

    @torch.no_grad()
    def mtm_sampling(self, trajectory: Dict[str, torch.Tensor], h):

        rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode = self.tokenizer_manager.encode(trajectory)
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, rcbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)
        sample_action = action_dist.sample()[0, self.cfg.traj_length - h]

        eval_action = action_dist.mean[0, self.cfg.traj_length - h]
        return sample_action, eval_action

    @torch.no_grad()
    def critic_guiding(self, trajectory: Dict[str, torch.Tensor], h):

        cur_state = trajectory["states"][:, self.cfg.traj_length - h, :]
        encode = self.tokenizer_manager.encode(trajectory)
        torch_rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode = self.tokenizer_manager.encode(trajectory)
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, torch_rcbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)
        sample_actions = action_dist.sample((1024,))[
            :, 0, self.cfg.traj_length - h, 0, :
        ]  # (30, act_dim)
        sample_states = cur_state.repeat(1024, 1)
        values = self.iql.qf(sample_states, sample_actions).squeeze(-1)
        values -= torch.max(values)
        values *= self.cfg.temperature
        p = torch.exp(values) / torch.exp(values).sum()
        # max_idx = torch.argmax(p)
        # eval_action = sample_actions[max_idx]
        eval_action = (sample_actions * p[:, None]).sum(dim=0) / p.sum()
        sample_idx = torch.multinomial(p, 1)
        sample_action = sample_actions[sample_idx]

        return sample_action, eval_action

    @torch.no_grad()
    def noise_adding(self, trajectory: Dict[str, torch.Tensor], h):

        cur_state = trajectory["states"][:, self.cfg.traj_length - h, :]
        encode = self.tokenizer_manager.encode(trajectory)
        torch_rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode = self.tokenizer_manager.encode(trajectory)
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, torch_rcbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)
        sample_actions = action_dist.mean[0, self.cfg.traj_length - h]
        noise = torch.randn_like(sample_actions) * 0.09
        sample_action = sample_actions + noise
        sample_action = torch.clamp(
            sample_action, -0.99999, 0.99999
        )  # TODO: check this
        eval_action = action_dist.mean[0, self.cfg.traj_length - h]
        sample_action = sample_actions[0]
        # print("adding noise")

        return sample_action, eval_action

    @torch.no_grad()
    def critic_lambda_guiding(
        self, trajectory: Dict[str, torch.Tensor], h: int, lmbda: float
    ):

        trajectory_batch = {
            k: v.repeat(self.cfg.action_samples, 1, 1) for k, v in trajectory.items()
        }
        encode = self.tokenizer_manager.encode(trajectory)
        torch_rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, torch_rcbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)
        sample_actions = action_dist.sample((self.cfg.action_samples,))[
            :, 0, self.cfg.traj_length - h :, 0, :
        ]  # (1024, h, act_dim)
        trajectory_batch["actions"][:, self.cfg.traj_length - h :, :] = sample_actions
        torch_fd_mask = create_fd_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode_batch = self.tokenizer_manager.encode(trajectory_batch)
        decode = self.tokenizer_manager.decode(self.mtm(encode_batch, torch_fd_mask))
        future_states = decode["states"][
            :, self.cfg.traj_length - h :, :
        ]  # (1024, h, state_dim)
        future_rewards = decode["rewards"][
            :, self.cfg.traj_length - h :, :
        ]  # (1024, h, 1)
        expect_return = torch.zeros((self.cfg.action_samples,), device=self.cfg.device)
        for t in range(h):
            values = torch.zeros(
                (self.cfg.action_samples, t + 1), device=self.cfg.device
            )
            discounts = torch.cumprod(
                self.cfg.discount * torch.ones((t + 1,), device=self.cfg.device), dim=0
            )
            if t > 0:
                values[:, :t] = future_rewards[:, :t, 0]
            values[:, t] = self.iql.qf(
                future_states[:, t], sample_actions[:, t]
            ).squeeze(-1)
            values *= discounts[None, :]
            if t < h - 1:
                expect_return += values.sum(dim=-1) * (1 - lmbda) * (lmbda**t)
            else:
                expect_return += values.sum(dim=-1) * (lmbda**t)

        expect_return -= torch.max(expect_return)
        score = expect_return * self.cfg.temperature
        p = torch.exp(score) / torch.exp(score).sum()
        # max_idx = torch.argmax(p)
        # eval_action = sample_actions[max_idx, 0]
        eval_action = (sample_actions[:, 0] * p[:, None]).sum(dim=0) / p.sum()
        sample_idx = torch.multinomial(p, 1)
        sample_action = sample_actions[sample_idx, 0]

        return sample_action, eval_action
    
    @torch.no_grad()
    def rtg_guiding(
        self, trajectory: Dict[str, torch.Tensor], h: int, lmbda: float=0.6
    ):
        
        trajectory_batch = {
            k: v.repeat(self.cfg.action_samples, 1, 1) for k, v in trajectory.items()
        }
        encode = self.tokenizer_manager.encode(trajectory)
        torch_rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, torch_rcbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)
        sample_actions = action_dist.sample((self.cfg.action_samples,))[
            :, 0, self.cfg.traj_length - h :, 0, :
        ]  # (1024, h, act_dim)
        trajectory_batch["actions"][:, self.cfg.traj_length - h :, :] = sample_actions
        torch_fd_mask = create_fd_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode_batch = self.tokenizer_manager.encode(trajectory_batch)
        decode = self.tokenizer_manager.decode(self.mtm(encode_batch, torch_fd_mask))
        future_states = decode["states"][
            :, self.cfg.traj_length - h :, :
        ]  # (1024, h, state_dim)
        future_rewards = decode["rewards"][
            :, self.cfg.traj_length - h :, :
        ]  # (1024, h, 1)
        expect_return = torch.zeros((self.cfg.action_samples,), device=self.cfg.device)
        for t in range(h):
            
            values = torch.zeros(
                (self.cfg.action_samples, t + 1), device=self.cfg.device
            )
            values[:, t] = decode["returns"][:, self.cfg.traj_length - h + t, 0]*1000
            
            discounts = torch.cumprod(
                self.cfg.discount * torch.ones((t + 1,), device=self.cfg.device), dim=0
            )
            if t > 0:
                values[:, :t] = future_rewards[:, :t, 0]
            values *= discounts[None, :]
            if t < h - 1:
                expect_return += values.sum(dim=-1) * (1 - lmbda) * (lmbda**t)
            else:
                expect_return += values.sum(dim=-1) * (lmbda**t)

        expect_return -= torch.max(expect_return)
        score = expect_return * 0.4
        p = torch.exp(score) / torch.exp(score).sum()
        # max_idx = torch.argmax(p)
        # eval_action = sample_actions[max_idx, 0]
        eval_action = (sample_actions[:, 0] * p[:, None]).sum(dim=0) / p.sum()
        sample_idx = torch.multinomial(p, 1)
        sample_action = sample_actions[sample_idx, 0]

        return sample_action, eval_action

        trajectory_batch = {
            k: v.repeat(self.cfg.action_samples, 1, 1) for k, v in trajectory.items()
        }
        encode = self.tokenizer_manager.encode(trajectory)
        torch_rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, torch_rcbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)
        sample_actions = action_dist.sample((self.cfg.action_samples,))[
            :, 0, self.cfg.traj_length - h :, 0, :
        ]  # (1024, h, act_dim)
        trajectory_batch["actions"][:, self.cfg.traj_length - h :, :] = sample_actions
        torch_ret_mask = create_ret_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode_batch = self.tokenizer_manager.encode(trajectory_batch)
        decode = self.tokenizer_manager.decode(self.mtm(encode_batch, torch_ret_mask))
        expect_return = decode["returns"][:, self.cfg.traj_length - h + 1, 0]
        expect_return -= torch.max(expect_return)
        score = expect_return * 10
        p = torch.exp(score) / torch.exp(score).sum()
        # max_idx = torch.argmax(p)
        # eval_action = sample_actions[max_idx, 0]
        eval_action = (sample_actions[:, 0] * p[:, None]).sum(dim=0) / p.sum()
        sample_idx = torch.multinomial(p, 1)
        sample_action = sample_actions[sample_idx, 0]

        return sample_action, eval_action

    @torch.no_grad()
    def action_sample(
        self,
        sequence_history,
        percentage=1.0,
        horizon=4,
        plan=True,
        eval=False,
        rtg=None,
    ):
        if eval == True:
            assert rtg is not None

        horizon = self.cfg.horizon
        end_idx = sequence_history["path_length"]
        if end_idx + horizon < self.cfg.traj_length:
            horizon = self.cfg.traj_length - end_idx
        obs_dim = sequence_history["observations"].shape[-1]
        action_dim = sequence_history["actions"].shape[-1]
        zero_trajectory = {
            "observations": np.zeros((1, self.cfg.traj_length, obs_dim)),
            "actions": np.zeros((1, self.cfg.traj_length, action_dim)),
            "rewards": np.zeros((1, self.cfg.traj_length, 1)),
            "values": np.zeros((1, self.cfg.traj_length, 1)),
        }
        history_length = self.cfg.traj_length - horizon + 1

        for k in zero_trajectory.keys():
            zero_trajectory[k][0, :history_length] = sequence_history[k][
                end_idx - history_length + 1 : end_idx + 1
            ]

        torch_zero_trajectory = {
            (
                "states" if k == "observations" else "returns" if k == "values" else k
            ): torch.tensor(v, device=self.cfg.device, dtype=torch.float32)
            for k, v in zero_trajectory.items()
        }

        if rtg is not None:
            # Use time-related rtg for eval
            return_to_go = float(rtg)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )
        else:
            # Use a constant rtg for explore
            return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
            return_min = self.tokenizer_manager.tokenizers["returns"].stats.min

            return_value = return_min + (return_max - return_min) * percentage
            return_to_go = float(return_value)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )

        # TODO: here we categrorize 'add noise' to planning, but actually it's not
        if plan:
            assert self.cfg.plan_guidance in [
                "critic_guiding",
                "critic_lambda_guiding",
                "noise_adding",
                "rtg_guiding",
            ]
            if self.cfg.plan_guidance == "critic_guiding":
                sample_action, eval_action = self.critic_guiding(
                    torch_zero_trajectory, horizon
                )

            elif self.cfg.plan_guidance == "critic_lambda_guiding":
                sample_action, eval_action = self.critic_lambda_guiding(
                    torch_zero_trajectory, horizon, lmbda=self.cfg.lmbda
                )

            elif self.cfg.plan_guidance == "noise_adding":
                sample_action, eval_action = self.noise_adding(
                    torch_zero_trajectory, horizon
                )
            elif self.cfg.plan_guidance == "rtg_guiding":
                sample_action, eval_action = self.rtg_guiding(
                    torch_zero_trajectory, horizon
                )

        else:
            sample_action, eval_action = self.mtm_sampling(
                torch_zero_trajectory, horizon
            )

        if eval:
            return eval_action
        else:
            return sample_action

    def compute_mtm_loss(
        self,
        batch: Dict[str, torch.Tensor],
        data_shapes,
        discrete_map,
        entropy_reg: float,
    ):

        # calculate future prediction loss
        losses = {}
        masked_losses = {}
        masked_c_losses = {}
        encoded_batch = self.tokenizer_manager.encode(batch)
        targets = encoded_batch

        masks = create_random_autoregressize_mask(
            data_shapes,
            self.cfg.mask_ratio,
            self.cfg.traj_length,
            self.cfg.device,
            self.cfg.p_weights,
        )
        preds = self.mtm(encoded_batch, masks)

        for key in targets.keys():
            target = targets[key]
            pred = preds[key]
            mask = masks[key]
            if len(mask.shape) == 1:
                # only along time dimension: repeat across the given dimension
                mask = mask[:, None].repeat(1, target.shape[2])
            elif len(mask.shape) == 2:
                pass

            batch_size, T, P, _ = target.size()
            if discrete_map[key]:
                raw_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
                ).unsqueeze(3)
            else:
                if key == "actions":
                    # sperate calc the action loss
                    raw_loss = (
                        nn.MSELoss(reduction="none")(pred.mean, target)
                        * mask[None, :, :, None]
                    )
                    losses[key] = raw_loss.mean(dim=(2, 3)).mean()

                    continue
                else:
                    # apply normalization
                    raw_loss = nn.MSELoss(reduction="none")(pred, target)
                    raw_loss = nn.MSELoss(reduction="none")(pred, target)
                    # here not taking the masked result, all the loss is calculated

            # raw_loss shape = [batch_size, T, P, 1]
            loss = raw_loss.mean(dim=(2, 3)).mean()

            masked_c_loss = (
                (raw_loss * mask[None, :, :, None]).sum(dim=(1, 2, 3)) / mask.sum()
            ).mean()
            masked_loss = (
                (raw_loss * (1 - mask[None, :, :, None])).sum(dim=(1, 2, 3))
                / (1 - mask).sum()
            ).mean()
            losses[key] = loss
            masked_c_losses[key] = masked_c_loss
            masked_losses[key] = masked_loss

            loss = torch.sum(torch.stack(list(losses.values())))

            a = targets["actions"].clip(-1 + 1e-6, 1 - 1e-6)
            a_hat_dist = preds["actions"]
            log_likelihood = a_hat_dist.log_likelihood(a)[
                :, ~masks["actions"].squeeze().to(torch.bool), :
            ].mean()
            entropy = a_hat_dist.entropy()[
                :, ~masks["actions"].squeeze().to(torch.bool), :
            ].mean()
            act_loss = -(log_likelihood + entropy_reg * entropy)
            losses["entropy"] = entropy
            losses["nll"] = -log_likelihood

            loss += act_loss

            return loss, losses, masked_losses, masked_c_losses, entropy

    def mtm_update(self, batch, data_shapes, discrete_map):
        loss, losses, masked_losses, masked_c_losses, entropy = self.compute_mtm_loss(
            batch, data_shapes, discrete_map, self.mtm.temperature().detach()
        )
        log_dict = {}
        for k, l in losses.items():
            log_dict[f"train/loss_{k}"] = l

            if k in masked_losses.keys():
                log_dict[f"train/masked_loss_{k}"] = masked_losses[k]
            if k in masked_c_losses.keys():
                log_dict[f"train/masked_c_loss_{k}"] = masked_c_losses[k]

        log_dict[f"train/loss"] = loss.item()
        log_dict["train/lr"] = self.mtm_scheduler.get_last_lr()[0]

        # backprop
        self.mtm.zero_grad(set_to_none=True)
        loss.backward()
        self.mtm_optimizer.step()
        self.mtm_scheduler.step()

        self.temp_optimizer.zero_grad()
        temperature_loss = (
            self.mtm.temperature() * (entropy - self.mtm.target_entropy).detach()
        )
        temperature_loss.backward()
        self.temp_optimizer.step()

        log_dict["train/temperature"] = self.mtm.temperature().item()
        log_dict["train/entropy"] = entropy.item()

        return log_dict

    def critic_update(self, experience: Tuple[torch.Tensor]):

        log_dict = self.iql.train(experience)

        return log_dict

    @torch.no_grad()
    def evaluate(
        self,
        num_episodes: int,
        episode_rtg_ref: np.ndarray,
        disable_tqdm: bool = True,
        verbose: bool = False,
        all_results: bool = False,
        num_videos: int = 3,
    ) -> Dict[str, Any]:

        log_data = {}

        for ratio in [0.9, 1.0]:
            stats: Dict[str, Any] = defaultdict(list)
            successes = None
            for i in range(num_episodes):
                current_trajectory = {
                    "observations": np.zeros(
                        (1000, self.env.observation_space.shape[0]), dtype=np.float32
                    ),
                    "actions": np.zeros(
                        (1000, self.env.action_space.shape[0]), dtype=np.float32
                    ),
                    "rewards": np.zeros((1000, 1), dtype=np.float32),
                    "values": np.zeros((1000, 1), dtype=np.float32),
                    "total_return": 0,
                    "path_length": 0,
                }

                observation, done = self.env.reset(), False
                # if len(videos) < num_videos:
                #     try:
                #         imgs = [self.env.sim.render(64, 48, camera_name="track")[::-1]]
                #     except:
                #         imgs = [self.env.render()[::-1]]

                timestep = 0
                while not done and timestep < 1000:
                    current_trajectory["observations"][timestep] = observation
                    action = self.action_sample(
                        current_trajectory,
                        percentage=1.0,
                        plan=False,
                        eval=True,
                        rtg=episode_rtg_ref[timestep] * ratio,
                    )
                    action = np.clip(action.cpu().numpy(), -1, 1)
                    new_observation, reward, done, info = self.env.step(action)
                    current_trajectory["actions"][timestep] = action
                    current_trajectory["rewards"][timestep] = reward
                    observation = new_observation
                    timestep += 1
                    current_trajectory["path_length"] += 1
                    # if len(videos) < num_videos:
                    #     try:
                    #         imgs.append(self.env.sim.render(64, 48, camera_name="track")[::-1])
                    #     except:
                    #         imgs.append(self.env.render()[::-1])

                # if len(videos) < num_videos:
                #     videos.append(np.array(imgs[:-1]))

                if "episode" in info:
                    for k in info["episode"].keys():
                        stats[k].append(float(info["episode"][k]))
                        if verbose:
                            print(f"{k}:{info['episode'][k]}")

                    ret = info["episode"]["return"]
                    mean = np.mean(stats["return"])
                    if "is_success" in info:
                        if successes is None:
                            successes = 0.0
                        successes += info["is_success"]

                else:
                    stats["return"].append(current_trajectory["rewards"].sum())
                    stats["length"].append(current_trajectory["path_length"])

            new_stats = {}
            for k, v in stats.items():
                new_stats[k + "_mean"] = float(np.mean(v))
                new_stats[k + "_std"] = float(np.std(v))

            if all_results:
                new_stats.update(stats)
            stats = new_stats
            if successes is not None:
                stats["success"] = successes / num_episodes
            for k, v in stats.items():
                log_data[f"eval_bc_{ratio}/{k}"] = v

            # for idx, v in enumerate(videos):
            #     log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
            #         v.transpose(0, 3, 1, 2), fps=10, format="gif"
            #     )
        print(stats["return_mean"])
        return log_data, stats["return_mean"]

    @torch.no_grad()
    def evaluate_plan(
        self,
        num_episodes: int,
        episode_rtg_ref: np.ndarray,
        disable_tqdm: bool = True,
        verbose: bool = False,
        all_results: bool = False,
        num_videos: int = 3,
    ) -> Dict[str, Any]:

        log_data = {}

        for ratio in [1.0]:
            stats: Dict[str, Any] = defaultdict(list)
            successes = None
            for i in range(num_episodes):
                current_trajectory = {
                    "observations": np.zeros(
                        (1000, self.env.observation_space.shape[0]), dtype=np.float32
                    ),
                    "actions": np.zeros(
                        (1000, self.env.action_space.shape[0]), dtype=np.float32
                    ),
                    "rewards": np.zeros((1000, 1), dtype=np.float32),
                    "values": np.zeros((1000, 1), dtype=np.float32),
                    "total_return": 0,
                    "path_length": 0,
                }

                observation, done = self.env.reset(), False
                # if len(videos) < num_videos:
                #     try:
                #         imgs = [self.env.sim.render(64, 48, camera_name="track")[::-1]]
                #     except:
                #         imgs = [self.env.render()[::-1]]

                timestep = 0
                while not done and timestep < 1000:
                    current_trajectory["observations"][timestep] = observation
                    action = self.action_sample(
                        current_trajectory,
                        percentage=1.0,
                        plan=True,
                        eval=True,
                        rtg=episode_rtg_ref[timestep] * ratio,
                    )
                    action = np.clip(action.cpu().numpy(), -1, 1)
                    new_observation, reward, done, info = self.env.step(action)
                    current_trajectory["actions"][timestep] = action
                    current_trajectory["rewards"][timestep] = reward
                    observation = new_observation
                    timestep += 1
                    current_trajectory["path_length"] += 1
                    # if len(videos) < num_videos:
                    #     try:
                    #         imgs.append(self.env.sim.render(64, 48, camera_name="track")[::-1])
                    #     except:
                    #         imgs.append(self.env.render()[::-1])

                # if len(videos) < num_videos:
                #     videos.append(np.array(imgs[:-1]))

                if "episode" in info:
                    for k in info["episode"].keys():
                        stats[k].append(float(info["episode"][k]))
                        if verbose:
                            print(f"{k}:{info['episode'][k]}")

                    ret = info["episode"]["return"]
                    mean = np.mean(stats["return"])
                    if "is_success" in info:
                        if successes is None:
                            successes = 0.0
                        successes += info["is_success"]

                else:
                    stats["return"].append(current_trajectory["rewards"].sum())
                    stats["length"].append(current_trajectory["path_length"])

            new_stats = {}
            for k, v in stats.items():
                new_stats[k + "_mean"] = float(np.mean(v))
                new_stats[k + "_std"] = float(np.std(v))

            if all_results:
                new_stats.update(stats)
            stats = new_stats
            if successes is not None:
                stats["success"] = successes / num_episodes
            for k, v in stats.items():
                log_data[f"eval_plan_{ratio}/{k}"] = v

            # for idx, v in enumerate(videos):
            #     log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
            #         v.transpose(0, 3, 1, 2), fps=10, format="gif"
            #     )
        print(stats["return_mean"])
        return log_data, stats["return_mean"]

    torch.no_grad()

    def evaluate_policy(
        self,
        num_episodes: int,
        disable_tqdm: bool = True,
        verbose: bool = False,
        all_results: bool = False,
        num_videos: int = 3,
    ) -> Dict[str, Any]:

        stats: Dict[str, Any] = defaultdict(list)
        successes = None

        pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)

        videos = []

        for i in pbar:

            observation, done = self.env.reset(), False
            # if len(videos) < num_videos:
            #     try:
            #         imgs = [self.env.sim.render(64, 48, camera_name="track")[::-1]]
            #     except:
            #         imgs = [self.env.render()[::-1]]

            timestep = 0
            while not done and timestep < 1000:
                action = self.iql.actor.act(observation, self.cfg.device)
                new_observation, reward, done, info = self.env.step(action)
                observation = new_observation
                timestep += 1
                # if len(videos) < num_videos:
                #     try:
                #         imgs.append(self.env.sim.render(64, 48, camera_name="track")[::-1])
                #     except:
                #         imgs.append(self.env.render()[::-1])

            # if len(videos) < num_videos:
            #     videos.append(np.array(imgs[:-1]))

            for k in info["episode"].keys():
                stats[k].append(float(info["episode"][k]))
                if verbose:
                    print(f"{k}:{info['episode'][k]}")

            ret = info["episode"]["return"]
            mean = np.mean(stats["return"])
            pbar.set_description(f"iter={i}\t last={ret:.2f} mean={mean}")
            if "is_success" in info:
                if successes is None:
                    successes = 0.0
                successes += info["is_success"]

        new_stats = {}
        for k, v in stats.items():
            new_stats[k + "_mean"] = float(np.mean(v))
            new_stats[k + "_std"] = float(np.std(v))

        if all_results:
            new_stats.update(stats)
        stats = new_stats
        if successes is not None:
            stats["success"] = successes / num_episodes

        log_data = {}
        for k, v in stats.items():
            log_data[f"eval_policy/{k}"] = v
        # for idx, v in enumerate(videos):
        #     log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
        #         v.transpose(0, 3, 1, 2), fps=10, format="gif"
        #     )

        return log_data, stats["return_mean"]


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
