from research.finetune.masks import *
from research.finetune.model import Critic, Value
from research.mtm.models.mtm_model import MTM
from research.mtm.tokenizers.base import TokenizerManager
from research.mtm.datasets.sequence_dataset import Trajectory
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


class Learner(object):
    def __init__(
        self,
        cfg,
        env: gym.Env,
        data_shapes,
        model_config,
        pretrain_model_path,
        pretrain_critic1_path,
        pretrain_critic2_path,
        pretrain_value_path,
        tokenizer_manager: TokenizerManager,
        discrete_map: Dict[str, torch.Tensor],
    ):
        self.cfg = cfg
        self.env = env
        self.mtm = model_config.create(data_shapes, cfg.traj_length, discrete_map)
        self.mtm.load_state_dict(torch.load(pretrain_model_path)["model"])
        self.mtm.to(cfg.device)
        self.critic1 = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.critic1.load_state_dict(torch.load(pretrain_critic1_path))
        self.critic1.to(cfg.device)
        self.critic2 = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.critic2.load_state_dict(torch.load(pretrain_critic2_path))
        self.critic2.to(cfg.device)
        self.critic1_target = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.critic1_target.load_state_dict(torch.load(pretrain_critic1_path))
        self.critic1_target.to(cfg.device)
        self.critic2_target = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.critic2_target.load_state_dict(torch.load(pretrain_critic2_path))
        self.critic2_target.to(cfg.device)
        self.value = Value(env.observation_space.shape[-1], cfg.critic_hidden_size)
        self.value.load_state_dict(torch.load(pretrain_value_path))
        self.value.to(cfg.device)
        self.tokenizer_manager = tokenizer_manager
        self.discrete_map = discrete_map
        self.mtm_optimizer = MTM.configure_optimizers(
            self.mtm,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        self.eps = 1e-5

        def _schedule(step):
            # warmp for 1000 steps
            if step < self.cfg.warmup_steps:
                return step / cfg.warmup_steps

            # then cosine decay
            assert self.cfg.num_train_steps > self.cfg.warmup_steps
            return 0.5 * (
                1 + np.cos(step / (cfg.num_train_steps - cfg.warmup_steps) * np.pi)
            )

        self.mtm_scheduler = LambdaLR(self.mtm_optimizer, lr_lambda=_schedule)

        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=self.cfg.v_lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )

    def bs_planning(self, trajectory: Dict[str, torch.Tensor], h: int):

        trajectories = {
            k: v.repeat(self.cfg.beam_width, 1, 1) for k, v in trajectory.items()
        }
        with torch.no_grad():
            for i in range(self.cfg.traj_length - h, self.cfg.traj_length):
                encode = self.tokenizer_manager.encode(trajectories)
                torch_rcbc_mask = create_rcbc_mask(
                    self.cfg.traj_length, self.cfg.device, i
                )
                copied_states = trajectories["states"][:, i, :].repeat_interleave(
                    self.cfg.action_samples, dim=0
                )  # [beam_width*n, obs_dim]
                if i > self.cfg.traj_length - h:
                    copied_cum_rewards = trajectories["rewards"][
                        :, self.cfg.traj_length - h : i, :
                    ].sum(dim=1)
                else:
                    # first step of planning, no rewards, only consider value
                    copied_cum_rewards = torch.zeros_like(
                        trajectories["rewards"][:, i, :]
                    )
                copied_cum_rewards = copied_cum_rewards.repeat_interleave(
                    self.cfg.action_samples, dim=0
                )  # [beam_width*n, 1]
                action_mean = self.tokenizer_manager.decode(
                    self.mtm(encode, torch_rcbc_mask)
                )["actions"][:, i, :]
                sampled_actions_mean = action_mean.repeat_interleave(
                    self.cfg.action_samples, dim=0
                )  # [beam_width*n, action_dim]
                sampled_actions = (
                    sampled_actions_mean
                    + self.cfg.action_noise_std
                    * torch.randn(
                        sampled_actions_mean.shape, device=sampled_actions_mean.device
                    )
                )
                expected_return = copied_cum_rewards + self.cfg.discount ** (
                    i - (self.cfg.traj_length - h)
                ) * torch.minimum(
                    self.critic1(copied_states, sampled_actions),
                    self.critic2(copied_states, sampled_actions),
                )
                sorted_return, sorted_idx = torch.sort(
                    expected_return.squeeze(-1), descending=True
                )
                trajectories["actions"][:, i, :] = sampled_actions[
                    sorted_idx[: self.cfg.beam_width]
                ]

                if i < self.cfg.traj_length - 1:
                    encode = self.tokenizer_manager.encode(trajectories)
                    torch_rew_mask = create_rew_mask(
                        self.cfg.traj_length, self.cfg.device, i
                    )
                    torch_fd_mask = create_fd_mask(
                        self.cfg.traj_length, self.cfg.device, i + 1
                    )
                    rew_pred = self.tokenizer_manager.decode(
                        self.mtm(encode, torch_rew_mask)
                    )["rewards"][:, i, :]
                    next_state_pred = self.tokenizer_manager.decode(
                        self.mtm(encode, torch_fd_mask)
                    )["states"][:, i + 1, :]
                    trajectories["rewards"][:, i, :] = rew_pred
                    trajectories["states"][:, i + 1, :] = next_state_pred

            max_return = sorted_return.max(0)[0]
            score = torch.exp(
                self.cfg.temperature
                * (sorted_return[: self.cfg.beam_width] - max_return)
            )
            sample_idx = torch.multinomial(score, 1)[0]
            sampled_action = trajectories["actions"][
                sample_idx, self.cfg.traj_length - h, :
            ]
            best_action = trajectories["actions"][0, self.cfg.traj_length - h, :]
            return sampled_action, best_action

    def filtered_uniform(self, trajectory: Dict[str, torch.Tensor]):

        trajectories = {
            k: v.repeat(self.cfg.beam_width, 1, 1) for k, v in trajectory.items()
        }
        with torch.no_grad():
            encode = self.tokenizer_manager.encode(trajectories)
            torch_rcbc_mask = create_rcbc_mask(
                self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - 1
            )
            copied_states = trajectories["states"][
                :, self.cfg.traj_length - 1, :
            ].repeat_interleave(
                self.cfg.action_samples, dim=0
            )  # [beam_width*n, obs_dim]
            # first step of planning, no rewards, only consider value
            copied_cum_rewards = torch.zeros_like(
                trajectories["rewards"][:, self.cfg.traj_length - 1, :]
            )
            copied_cum_rewards = copied_cum_rewards.repeat_interleave(
                self.cfg.action_samples, dim=0
            )  # [beam_width*n, 1]
            action_mean = self.tokenizer_manager.decode(
                self.mtm(encode, torch_rcbc_mask)
            )["actions"][:, self.cfg.traj_length - 1, :]
            sampled_actions_mean = action_mean.repeat_interleave(
                self.cfg.action_samples, dim=0
            )  # [beam_width*n, action_dim]
            sampled_actions = (
                sampled_actions_mean
                + self.cfg.action_noise_std
                * torch.randn(
                    sampled_actions_mean.shape, device=sampled_actions_mean.device
                )
            )
            expected_return = copied_cum_rewards + torch.minimum(
                self.critic1(copied_states, sampled_actions),
                self.critic2(copied_states, sampled_actions),
            )
            sorted_return, sorted_idx = torch.sort(
                expected_return.squeeze(-1), descending=True
            )
            trajectories["actions"][:, self.cfg.traj_length - 1, :] = sampled_actions[
                sorted_idx[: self.cfg.beam_width]
            ]

            max_return = sorted_return.max(0)[0]
            # uniform sample from beam_width
            score = torch.ones(self.cfg.beam_width, device=self.cfg.device)
            sample_idx = torch.multinomial(score, 1)[0]
            sampled_action = trajectories["actions"][
                sample_idx, self.cfg.traj_length - 1, :
            ]
            best_action = trajectories["actions"][0, self.cfg.traj_length - 1, :]
            return sampled_action, best_action

    def critic_planning(self, trajectory: Dict[str, torch.Tensor]):

        trajectories = {
            k: v.repeat(self.cfg.beam_width, 1, 1) for k, v in trajectory.items()
        }
        with torch.no_grad():
            encode = self.tokenizer_manager.encode(trajectories)
            torch_rcbc_mask = create_rcbc_mask(
                self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - 1
            )
            copied_states = trajectories["states"][
                :, self.cfg.traj_length - 1, :
            ].repeat_interleave(self.cfg.action_samples, dim=0)

            copied_states += self.cfg.critic_noise_std * torch.randn(
                copied_states.shape, device=copied_states.device
            )
            # [beam_width*n, obs_dim]
            # first step of planning, no rewards, only consider value
            copied_cum_rewards = torch.zeros_like(
                trajectories["rewards"][:, self.cfg.traj_length - 1, :]
            )
            copied_cum_rewards = copied_cum_rewards.repeat_interleave(
                self.cfg.action_samples, dim=0
            )  # [beam_width*n, 1]
            action_mean = self.tokenizer_manager.decode(
                self.mtm(encode, torch_rcbc_mask)
            )["actions"][:, self.cfg.traj_length - 1, :]
            sampled_actions_mean = action_mean.repeat_interleave(
                self.cfg.action_samples, dim=0
            )  # [beam_width*n, action_dim]
            sampled_actions = (
                sampled_actions_mean
                + self.cfg.action_noise_std
                * torch.randn(
                    sampled_actions_mean.shape, device=sampled_actions_mean.device
                )
            )
            expected_return = copied_cum_rewards + torch.minimum(
                self.critic1(copied_states, sampled_actions),
                self.critic2(copied_states, sampled_actions),
            )
            sorted_return, sorted_idx = torch.sort(
                expected_return.squeeze(-1), descending=True
            )
            trajectories["actions"][:, self.cfg.traj_length - 1, :] = sampled_actions[
                sorted_idx[: self.cfg.beam_width]
            ]

            max_return = sorted_return.max(0)[0]
            # uniform sample from beam_width
            score = torch.exp(
                self.cfg.temperature
                * (sorted_return[: self.cfg.beam_width] - max_return)
            )
            sample_idx = torch.multinomial(score, 1)[0]
            sampled_action = trajectories["actions"][
                sample_idx, self.cfg.traj_length - 1, :
            ]
            best_action = trajectories["actions"][0, self.cfg.traj_length - 1, :]
            return sampled_action, best_action

    def action_sample(
        self, sequence_history, percentage=1.0, horizon=4, plan=True, eval=False
    ):
        if eval == True:
            assert plan == False

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

        return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
        return_min = self.tokenizer_manager.tokenizers["returns"].stats.min

        return_value = return_min + (return_max - return_min) * percentage
        return_to_go = float(return_value)
        returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
        torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(self.cfg.device)

        if plan:
            assert self.cfg.plan_guidance in [
                "mtm_critic",
                "critic_filter",
                "critic_disturb",
            ]
            if self.cfg.plan_guidance == "mtm_critic":
                sample, best = self.bs_planning(torch_zero_trajectory, horizon)
            elif self.cfg.plan_guidance == "critic_filter":
                sample, best = self.filtered_uniform(torch_zero_trajectory)
            elif self.cfg.plan_guidance == "critic_disturb":
                sample, best = self.critic_planning(torch_zero_trajectory)
            return sample, best
        else:
            with torch.no_grad():
                encode = self.tokenizer_manager.encode(torch_zero_trajectory)
                torch_rcbc_mask = create_rcbc_mask(
                    self.cfg.traj_length,
                    self.cfg.device,
                    self.cfg.traj_length - horizon,
                )
                policy_action = self.tokenizer_manager.decode(
                    self.mtm(encode, torch_rcbc_mask)
                )["actions"][0, self.cfg.traj_length - horizon, :]
                if not eval:
                    policy_action += self.cfg.exploration_noise_std * torch.randn(
                        policy_action.shape, device=policy_action.device
                    )
            return policy_action, None

    def compute_mtm_loss(
        self, batch: Dict[str, torch.Tensor], data_shapes, discrete_map
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

            if discrete_map[key]:
                raw_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
                ).unsqueeze(3)
            else:
                # apply normalization
                if self.mtm.norm == "l2":
                    target = target / torch.norm(target, dim=-1, keepdim=True)
                elif self.mtm.norm == "mae":
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target_s = (target - mean) / (var + 1.0e-6) ** 0.5

                raw_loss = nn.MSELoss(reduction="none")(pred, target)

            # raw_loss shape = [batch_size, T, P, 1]
            loss = raw_loss.mean(dim=(2, 3)).mean()
            masked_c_loss = (
                (raw_loss * mask[None, :, :, None]).sum(dim=(1, 2, 3)) / mask.sum()
            ).mean()
            masked_loss = (
                (raw_loss * (1 - mask[None, :, :, None])).sum(dim=(1, 2, 3))
                / (1 - mask).sum()
            ).mean()

            if self.cfg.use_masked_loss:
                losses[key] = masked_loss
            else:
                losses[key] = loss
            masked_c_losses[key] = masked_c_loss
            masked_losses[key] = masked_loss

        if self.cfg.loss_weight is None:
            loss = torch.sum(torch.stack(list(losses.values())))
        else:
            loss = torch.sum(
                torch.stack(
                    [
                        losses[key] * weight
                        for key, weight in self.cfg.loss_weight.items()
                    ]
                )
            )

        return loss, losses, masked_losses, masked_c_losses

    def compute_q_loss(self, experience):

        states, actions, rewards, next_states, dones = experience

        with torch.no_grad():
            next_v = self.value(next_states) * (1 - dones)
            target_q_values = rewards + self.cfg.discount * next_v

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = ((q1 - target_q_values) ** 2).mean()
        critic2_loss = ((q2 - target_q_values) ** 2).mean()

        return critic1_loss, critic2_loss

    def compute_value_loss(self, experience):

        states, actions, rewards, next_states, dones = experience

        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2) * (1 - dones)

        value = self.value(states)
        value_loss = loss(min_Q - value, 0.8).mean()

        return value_loss

    def mtm_update(self, batch, data_shapes, discrete_map):
        loss, losses, masked_losses, masked_c_losses = self.compute_mtm_loss(
            batch, data_shapes, discrete_map
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

        return log_dict

    def critic_update(self, experience: Tuple[torch.Tensor]):
        critic1_loss, critic2_loss = self.compute_q_loss(experience)

        log_dict = {}

        log_dict[f"train/q1_loss"] = critic1_loss.item()
        log_dict[f"train/q2_loss"] = critic2_loss.item()

        # backprop
        self.critic1.zero_grad(set_to_none=True)
        self.critic2.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic2_loss.backward()

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return log_dict

    def value_update(self, experience: Tuple[torch.Tensor]):

        value_loss = self.compute_value_loss(experience)

        log_dict = {}
        log_dict[f"train/v_loss"] = value_loss
        self.value.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        return log_dict

    def critic_target_soft_update(self):

        for target_param, param in zip(
            self.critic1_target.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic2_target.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
            )

    def evaluate(
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
                action, _ = self.action_sample(
                    current_trajectory, percentage=1.0, plan=False, eval=True
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
                pbar.set_description(f"iter={i}\t last={ret:.2f} mean={mean}")
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
        print(stats["return_mean"])
        if successes is not None:
            stats["success"] = successes / num_episodes

        log_data = {}
        for k, v in stats.items():
            log_data[f"eval_bc/{k}"] = v
        # for idx, v in enumerate(videos):
        #     log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
        #         v.transpose(0, 3, 1, 2), fps=10, format="gif"
        #     )

        return log_data


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
