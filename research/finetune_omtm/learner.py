from research.finetune_omtm.masks import *
from research.finetune_omtm.model import Critic, Value, Actor
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
        pretrain_actor_path,
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
        self.critic2 = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.critic1_target = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.critic2_target = Critic(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        self.value = Value(env.observation_space.shape[-1], cfg.critic_hidden_size)
        self.actor = Actor(env.observation_space.shape[-1],
            env.action_space.shape[-1],
            cfg.critic_hidden_size,
        )
        if self.cfg.critic_scratch is not True:
            self.critic1.load_state_dict(torch.load(pretrain_critic1_path))
            self.critic2.load_state_dict(torch.load(pretrain_critic2_path))
            self.critic1_target.load_state_dict(torch.load(pretrain_critic1_path))
            self.critic2_target.load_state_dict(torch.load(pretrain_critic2_path))
            self.value.load_state_dict(torch.load(pretrain_value_path))
            self.actor.load_state_dict(torch.load(pretrain_actor_path))

        self.critic2.to(cfg.device)
        self.critic1.to(cfg.device)
        self.critic1_target.to(cfg.device)
        self.critic2_target.to(cfg.device)
        self.value.to(cfg.device)
        self.actor.to(cfg.device)

        self.tokenizer_manager = tokenizer_manager
        self.discrete_map = discrete_map
        self.mtm_optimizer = omtm.configure_optimizers(
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
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.cfg.v_lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        self.temp_optimizer = torch.optim.Adam(
            [self.mtm.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    # def bs_planning(self, trajectory: Dict[str, torch.Tensor], h: int):

    #     trajectories = {
    #         k: v.repeat(self.cfg.beam_width, 1, 1) for k, v in trajectory.items()
    #     }
    #     with torch.no_grad():
    #         for i in range(self.cfg.traj_length - h, self.cfg.traj_length):
    #             encode = self.tokenizer_manager.encode(trajectories)
    #             torch_rcbc_mask = create_rcbc_mask(
    #                 self.cfg.traj_length, self.cfg.device, i
    #             )
    #             copied_states = trajectories["states"][:, i, :].repeat_interleave(
    #                 self.cfg.action_samples, dim=0
    #             )  # [beam_width*n, obs_dim]
    #             if i > self.cfg.traj_length - h:
    #                 copied_cum_rewards = trajectories["rewards"][
    #                     :, self.cfg.traj_length - h : i, :
    #                 ].sum(dim=1)
    #             else:
    #                 # first step of planning, no rewards, only consider value
    #                 copied_cum_rewards = torch.zeros_like(
    #                     trajectories["rewards"][:, i, :]
    #                 )
    #             copied_cum_rewards = copied_cum_rewards.repeat_interleave(
    #                 self.cfg.action_samples, dim=0
    #             )  # [beam_width*n, 1]
    #             action_mean = self.tokenizer_manager.decode(
    #                 self.mtm(encode, torch_rcbc_mask)
    #             )["actions"][:, i, :]
    #             sampled_actions_mean = action_mean.repeat_interleave(
    #                 self.cfg.action_samples, dim=0
    #             )  # [beam_width*n, action_dim]
    #             sampled_actions = (
    #                 sampled_actions_mean
    #                 + self.cfg.action_noise_std
    #                 * torch.randn(
    #                     sampled_actions_mean.shape, device=sampled_actions_mean.device
    #                 )
    #             )
    #             expected_return = copied_cum_rewards + self.cfg.discount ** (
    #                 i - (self.cfg.traj_length - h)
    #             ) * torch.minimum(
    #                 self.critic1(copied_states, sampled_actions),
    #                 self.critic2(copied_states, sampled_actions),
    #             )
    #             sorted_return, sorted_idx = torch.sort(
    #                 expected_return.squeeze(-1), descending=True
    #             )
    #             trajectories["actions"][:, i, :] = sampled_actions[
    #                 sorted_idx[: self.cfg.beam_width]
    #             ]

    #             if i < self.cfg.traj_length - 1:
    #                 encode = self.tokenizer_manager.encode(trajectories)
    #                 torch_rew_mask = create_rew_mask(
    #                     self.cfg.traj_length, self.cfg.device, i
    #                 )
    #                 torch_fd_mask = create_fd_mask(
    #                     self.cfg.traj_length, self.cfg.device, i + 1
    #                 )
    #                 rew_pred = self.tokenizer_manager.decode(
    #                     self.mtm(encode, torch_rew_mask)
    #                 )["rewards"][:, i, :]
    #                 next_state_pred = self.tokenizer_manager.decode(
    #                     self.mtm(encode, torch_fd_mask)
    #                 )["states"][:, i + 1, :]
    #                 trajectories["rewards"][:, i, :] = rew_pred
    #                 trajectories["states"][:, i + 1, :] = next_state_pred

    #         max_return = sorted_return.max(0)[0]
    #         score = torch.exp(
    #             self.cfg.temperature
    #             * (sorted_return[: self.cfg.beam_width] - max_return)
    #         )
    #         sample_idx = torch.multinomial(score, 1)[0]
    #         sampled_action = trajectories["actions"][
    #             sample_idx, self.cfg.traj_length - h, :
    #         ]
    #         best_action = trajectories["actions"][0, self.cfg.traj_length - h, :]
    #         return sampled_action, best_action

    @torch.no_grad()
    def mtm_sampling(self, trajectory: Dict[str, torch.Tensor], h):
        
        rcbc_mask = create_rcbc_mask(self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h)
        encode = self.tokenizer_manager.encode(trajectory)
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, rcbc_mask))['actions'] # dist of shape(1, seq_len, act_dim)
        sample_action = action_dist.sample()[0, self.cfg.traj_length - h]
        
        mean_action = action_dist.mean[0, self.cfg.traj_length - h]
        return sample_action, mean_action
        
        
    @torch.no_grad()
    def critic_guiding(self, trajectory: Dict[str, torch.Tensor], h):

        cur_state = trajectory["states"][:, self.cfg.traj_length - h, :]
        encode = self.tokenizer_manager.encode(trajectory)
        torch_rcbc_mask = create_rcbc_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - h
        )
        encode = self.tokenizer_manager.encode(trajectory)
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, torch_rcbc_mask))["actions"] # dist of shape(1, seq_len, act_dim)
        sample_actions = action_dist.sample((30,))[:, 0, self.cfg.traj_length - h, 0, :] #(30, act_dim)
        sample_states = cur_state.repeat(30, 1)
        values = torch.min(self.critic1(sample_states, sample_actions), self.critic2(sample_states, sample_actions)).squeeze(-1)
        values -= torch.max(values)
        p = torch.exp(values) / torch.exp(values).sum()
        sample_idx = torch.multinomial(p, 1)
        sample_action = sample_actions[sample_idx]
        max_idx = p.max(dim=0)[1]
        best_action = sample_actions[max_idx]
        
        return sample_action, best_action
        
    @torch.no_grad()
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
                "critic_guiding",
            ]
            sample, best = self.critic_guiding(torch_zero_trajectory, horizon)
            return sample, best
        else:  
            if not eval:
                policy_action, _ = self.mtm_sampling(torch_zero_trajectory, horizon)
            else:
                _, policy_action = self.mtm_sampling(torch_zero_trajectory, horizon)
                
            return policy_action, None

    def compute_mtm_loss(
        self, batch: Dict[str, torch.Tensor], data_shapes, discrete_map, entropy_reg: float,
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
                    raw_loss = nn.MSELoss(reduction="none")(pred.mean, target) * mask[None, :, :, None]
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
        
        a = targets["actions"]
        a_hat_dist = preds["actions"]
        log_likelihood = a_hat_dist.log_likelihood(a)[:, ~masks['actions'].squeeze().to(torch.bool), :].mean()
        entropy = a_hat_dist.entropy()[:, ~masks['actions'].squeeze().to(torch.bool), :].mean()
        act_loss = -(log_likelihood + entropy_reg * entropy)
        losses['entropy'] = entropy
        losses['nll'] = - log_likelihood
        
        loss += act_loss
        
        return loss, losses, masked_losses, masked_c_losses, entropy

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
    
    def compute_policy_loss(self, experience):
        
        states, actions, rewards, next_states, dones = experience
        with torch.no_grad():
            v = self.value(states)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1,q2)

        exp_a = torch.exp((min_Q - v) * 3)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device))

        _, dist = self.actor.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss

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
        temperature_loss = self.mtm.temperature() * (entropy - self.mtm.target_entropy).detach()
        temperature_loss.backward()
        self.temp_optimizer.step()
        
        log_dict["train/temperature"] = self.mtm.temperature().item()
        log_dict["train/entropy"] = entropy.item()

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
    
    def policy_update(self, experience: Tuple[torch.Tensor]):
        
        policy_loss = self.compute_policy_loss(experience)
        
        log_dict = {}
        log_dict[f"train/policy_loss"] = policy_loss
        self.actor.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()

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

        return_to_go_list = [0.8, 0.9, 1.0]
        log_data = {}

        for return_to_go in return_to_go_list:
            
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
                    action, _ = self.action_sample(
                        current_trajectory, percentage=return_to_go, plan=False, eval=True
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
            print(stats["return_mean"])
            if successes is not None:
                stats["success"] = successes / num_episodes

            for k, v in stats.items():
                log_data[f"eval_bc_{return_to_go}/{k}"] = v
            # for idx, v in enumerate(videos):
            #     log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
            #         v.transpose(0, 3, 1, 2), fps=10, format="gif"
            #     )

        return log_data
    
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
                obs = torch.from_numpy(observation).float().to(self.cfg.device)
                action = self.actor.get_det_action(obs)
                action = action.numpy()
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
        print(stats["return_mean"])
        if successes is not None:
            stats["success"] = successes / num_episodes

        log_data = {}
        for k, v in stats.items():
            log_data[f"eval_policy/{k}"] = v
        # for idx, v in enumerate(videos):
        #     log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
        #         v.transpose(0, 3, 1, 2), fps=10, format="gif"
        #     )

        return log_data


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
