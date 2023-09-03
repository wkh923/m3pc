from research.finetune.finetune import RunConfig
from research.finetune.masks import *
from research.finetune.replay_buffer import ReplayBuffer
from research.mtm.models.mtm_model import MTM
from research.mtm.tokenizers.base import TokenizerManager
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR



class Learner(object):
    def __init__(self,
                 cfg: RunConfig,
                 model: MTM,
                 critic_model,
                 tokenizer_manager: TokenizerManager,
                 buffer: ReplayBuffer,
                 discrete_map: Dict[str, torch.Tensor]
                 ):
        self.cfg = cfg
        self.mtm = model
        self.critic_model = critic_model
        self.critic_target = critic_model
        self.tokenizer_manager = tokenizer_manager
        self.buffer = buffer
        self.discrete_map = discrete_map
        self.mtm_optimizer = MTM.configure_optimizers(
            self.mtm,
            learning_rate = self.cfg.learning_rate,
            weight_decay = self.cfg.weight_decay,
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
    
    def compute_mtm_loss(self, batch: Dict[str: torch.Tensor]):
        
        action_dim = batch["actions"].shape[-1]
        
        def compute_target_cem_action(trajectory: Dict[str, torch.Tensor]):
            
            cem_init_mean = self.tokenizer_manager.tokenizers["actions"].stats.mean
            cem_init_std = self.tokenizer_manager.tokenizers["actions"].stats.std
            cem_init_mean = torch.tensor(cem_init_mean, device=self.cfg.device).unsqueeze(0)
            cem_init_std = torch.tensor(cem_init_std, device=self.cfg.device).unsqueeze(0)
            with torch.no_grad():
                for it in range(self.cfg.n_iter):
                    
                    #Generate action samples
                    action_rsamples = cem_init_mean + cem_init_std * torch.randn(self.cfg.n_rsamples, action_dim, device=self.cfg.device)
                    encode = self.tokenizer_manager.encode(trajectory)
                    torch_rcbc_mask = create_rcbc_mask(self.cfg.traj_length, self.cfg.device)
                    pred = self.mtm(encode, torch_rcbc_mask)
                    action_policy = self.tokenizer_manager.decode(pred)["actions"][0, 0]
                    action_policy_samples = action_policy + self.cfg.policy_std * torch.randn(self.cfg.n_policy_samples, action_dim, device=self.cfg.device)
                    
                    action_samples = torch.cat([action_rsamples, action_policy_samples], dim=0)
                    
                    #Use the model to predict future sequence
                    sample_batch = {k: v.repeat(action_samples.shape[0], 1, 1) for k, v in trajectory.items()}
                    sample_batch["actions"][:, 0, :] = action_samples
                    
                    encode_batch = self.tokenizer_manager.encode(sample_batch)
                    torch_future_prediction_mask = create_future_prediction_mask(self.cfg.traj_length, self.cfg.device)
                    pred = self.mtm(encode_batch, torch_future_prediction_mask)
                    decode = self.tokenizer_manager.decode(pred)
                    future_rewards = decode["rewards"][:, :-1, :].squeeze #(num_samples, traj_length - 1)
                    discounts = torch.tensor([self.cfg.discount ** i for i in range(future_rewards.shape[1])], device=self.cfg.device)[None, :]
                    
                    expected_return = (future_rewards * discounts).sum(dim=1) \
                        + self.critic_model(decode["states"][:, -1, :], decode["actions"][:, -1, :]) * (self.cfg.discount ** future_rewards.shape[1])
                    
                    _, sorted_indices = torch.sort(expected_return, descending=True)
                    top_k_actions = action_samples[sorted_indices[:self.cfg.top_k]]
                    
                    cem_init_mean = torch.mean(top_k_actions, dim=0)
                    cem_init_std = torch.std(top_k_actions, dim=0, unbiased=False)
            
            return cem_init_mean, cem_init_std
        
        #calculate future prediction loss
        losses = {}
        masked_losses = {}
        masked_c_losses = {}
        encoded_batch = self.tokenizer_manager.encode(batch)
        targets = encoded_batch
        torch_future_prediction_mask = create_future_prediction_mask(self.cfg.traj_length, self.cfg.device)
        preds = self.mtm(encoded_batch, torch_future_prediction_mask)
        
        for key in targets.keys():
            target = targets[key]
            pred = preds[key]
            mask = torch_future_prediction_mask[key]
            
            if len(mask.shape) == 1:
                # only along time dimension: repeat across the given dimension
                mask = mask[:, None].repeat(1, target.shape[2])
            elif len(mask.shape) == 2:
                pass
            
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
        
        #calculate policy loss
        policy_loss = 0
        torch_rcbc_mask = create_rcbc_mask(self.cfg.traj_length, self.cfg.device)
        pred = self.mtm(encoded_batch, torch_rcbc_mask)
        actions = self.tokenizer_manager.decode(pred)["actions"][:, 0, :]
        batch_size = actions.shape[0]
        for traj_indx in range(batch_size):
            trajectory = {k:v[traj_indx].unsqueeze(0) for k, v in batch.items()}
            target_mean, target_std = compute_target_cem_action(trajectory)
            policy_loss += ((actions[traj_indx] - target_mean.squeeze(0)) ** 2) / (target_std.squeeze(0) ** 2)
        policy_loss /= batch_size
        
        losses["policy"] = policy_loss
        if self.cfg.loss_weight is None:
            loss = torch.sum(torch.stack(list(losses.values())))
        else:
            loss = torch.sum(torch.stack([losses[key] * weight for key, weight in self.cfg.loss_weight.items()]))
        
        return loss, losses, masked_losses, masked_c_losses
    
    def compute_q_loss(self, batch):
        
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
        predicted_q_values = self.critic_model(states, actions)
    
        next_actions = self.action_sample(next_states, actions.shape[-1], 1)
        target_next_q = self.critic_target(next_states, next_actions)
        target_q_values = rewards + self.cfg.discount * target_next_q
        
        q_loss = nn.MSELoss(predicted_q_values, target_q_values)
        
        return q_loss
    
    def mtm_update(self, batch):
        loss, losses, masked_losses, masked_c_losses = self.compute_mtm_loss(batch)
        log_dict = {}
        for key, loss in losses.items():
            log_dict[f"train/loss_{key}"] = loss
            if key in masked_losses.keys():
                log_dict[f"train/masked_loss_{key}"] = masked_losses[key]
            if key in masked_c_losses.keys():
                log_dict[f"train/masked_c_loss_{key}"] = masked_c_losses[key]
        log_dict[f"train/loss"] = loss.item()
        log_dict["train/lr"] = self.mtm_scheduler.get_last_lr()[0]
        
        #backprop
        self.mtm.zero_grad(set_to_none=True)
        loss.backward()
        self.mtm_optimizer.step()
        self.mtm_scheduler.step()
        
        return log_dict
        
    def critic_update(self,
                      batch: Dict[str, torch.Tensor]):
        critic_loss = self.compute_q_loss(batch)
        
        log_dict = {}
        log_dict[f"train/q_loss"] = critic_loss.item()
        
        #backprop
        self.critic_model.zero_grad(set_to_none=True)
        critic_loss.backward()
        
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        
        return log_dict
        
    def critic_target_soft_update(self):
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
    
    
    
    
    def action_sample(self, observations, action_dim, percentage=1.0):
        zero_trajectory = {
            "states": np.zeros((observations.shape[0], self.cfg.traj_length, observations.shape[-1])),
            "actions": np.zeros((observations.shape[0], self.cfg.traj_length, action_dim)),
            "rewards": np.zeros((observations.shape[0], self.cfg.traj_length, 1)),
            "returns": np.zeros((observations.shape[0], self.cfg.traj_length, 1))
        }
        torch_zero_trajectories = {
            k: torch.tensor(v, device=self.cfg.device) for k, v in zero_trajectory.items()}
        torch_zero_trajectories["states"][:, 0] = torch.tensor(observations)
        return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
        return_min = self.tokenizer_manager.tokenizers["returns"].stats.min
        
        return_value = return_min + (return_max - return_min) * percentage
        return_to_go = float(return_value)
        returns = return_to_go * np.ones((observations.shape[0], self.traj_length, 1))
        torch_zero_trajectories["returns"] = torch.tensor(returns)
        
        torch_rcbc_mask = create_rcbc_mask(self.cfg.traj_length, self.cfg.device)
        encode = self.tokenizer_manager.encode(torch_zero_trajectories)
        with torch.no_grad():
            pred = self.mtm(encode, torch_rcbc_mask)
        actions = self.tokenizer_manager.decode(pred)["actions"][:, 0, :].cpu().np()
        
        return actions    
    
    def evaluate(self,
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