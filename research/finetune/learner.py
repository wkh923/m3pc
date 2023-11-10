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
    def __init__(self,
                 cfg,
                 env: gym.Env,
                 data_shapes,
                 model_config,
                 pretrain_model_path,
                 pretrain_critic1_path,
                 pretrain_critic2_path,
                 pretrain_value_path,
                 tokenizer_manager: TokenizerManager,
                 discrete_map: Dict[str, torch.Tensor]
                 ):
        self.cfg = cfg
        self.env = env
        self.mtm = model_config.create(data_shapes, cfg.traj_length, discrete_map)
        self.mtm.load_state_dict(torch.load(pretrain_model_path)["model"])
        self.mtm.to(cfg.device)
        self.critic1 = Critic(env.observation_space.shape[-1], env.action_space.shape[-1], cfg.critic_hidden_size).to(cfg.device)
        self.critic1.load_state_dict(torch.load(pretrain_critic1_path))
        self.critic2 = Critic(env.observation_space.shape[-1], env.action_space.shape[-1], cfg.critic_hidden_size).to(cfg.device)
        self.critic2.load_state_dict(torch.load(pretrain_critic2_path))
        self.critic1_target = Critic(env.observation_space.shape[-1], env.action_space.shape[-1], cfg.critic_hidden_size).to(cfg.device)
        self.critic1_target.load_state_dict(torch.load(pretrain_critic1_path))
        self.critic2_target = Critic(env.observation_space.shape[-1], env.action_space.shape[-1], cfg.critic_hidden_size).to(cfg.device)
        self.critic2_target.load_state_dict(torch.load(pretrain_critic2_path))
        self.value = Value(env.observation_space.shape[-1], cfg.critic_hidden_size).to(cfg.device)
        self.value.load_state_dict(torch.load(pretrain_value_path))
        self.tokenizer_manager = tokenizer_manager
        self.discrete_map = discrete_map
        self.mtm_optimizer = MTM.configure_optimizers(
            self.mtm,
            learning_rate = self.cfg.learning_rate,
            weight_decay = self.cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        self.eps = 1e-5
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
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),
                                                 lr=self.cfg.critic_lr,
                                                 weight_decay = self.cfg.weight_decay,
                                                 betas=(0.9, 0.999))
        
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                 lr=self.cfg.critic_lr,
                                                 weight_decay = self.cfg.weight_decay,
                                                 betas=(0.9, 0.999))
        
        self.value_optimizer = torch.optim.Adam(self.value.parameters(),
                                                 lr=self.cfg.v_lr,
                                                 weight_decay = self.cfg.weight_decay,
                                                 betas=(0.9, 0.999))
    
    def compute_target_cem_action(self, trajectory: Dict[str, torch.Tensor], seg_idx: int):
        
        with torch.no_grad():
            action_values = self.tokenizer_manager.tokenizers["actions"].values[0,0,0,:]
            encode = self.tokenizer_manager.encode(trajectory)
            encode_batch = {k: v.repeat(self.cfg.n_policy_samples + self.cfg.n_rsamples, 1, 1, 1) for k, v in encode.items()}
            torch_rcbc_mask = create_rcbc_mask(traj_length=self.cfg.traj_length, device=self.cfg.device, pos=seg_idx)
            torch_fd_mask = create_fd_mask(traj_length=self.cfg.traj_length, device=self.cfg.device, pos=seg_idx)
            policy_pred = self.mtm(encode_batch, torch_rcbc_mask)["actions"][0, seg_idx:, :, :] #(traj_length-seg_idx+1, action_dim, num_bins)
            policy_dist = D.categorical.Categorical(logits=policy_pred)
            cem_dist = D.categorical.Categorical(logits=policy_pred)
            for it in range (self.cfg.n_iter):
                #Generate action samples
                action_policy_samples = policy_dist.sample((self.cfg.n_policy_samples,))
                action_rsamples = cem_dist.sample((self.cfg.n_rsamples,))
                action_samples = torch.cat([action_rsamples, action_policy_samples], dim=0)
                encode_action_samples = F.one_hot(action_samples, action_values.shape[0])
                 
                 
                #Use the model to predict future sequence
                encode_batch["actions"][:, seg_idx:, :, :] = encode_action_samples
                fd_pred = self.mtm(encode_batch, torch_fd_mask)
                decode = self.tokenizer_manager.decode(fd_pred)
                future_rewards = decode["rewards"][:, seg_idx:-1, :].squeeze(-1) #(num_samples, traj_length - seg_idx)
                discounts = torch.tensor([self.cfg.discount ** i for i in range(future_rewards.shape[1])], device=self.cfg.device)[None, :]
                expected_return = (future_rewards * discounts).sum(dim=1) + (self.value(decode["states"][:, -1, :]) * (self.cfg.discount ** future_rewards.shape[1])).squeeze(-1)
                
                sorted_return, sorted_indices = torch.sort(expected_return, descending=True)
                top_k_actions = action_samples[sorted_indices[:self.cfg.top_k]]
                encode_top_k_actions = F.one_hot(top_k_actions, action_values.shape[0]) # (k, traj_length-seg_idx+1, action_dim, num_bins)
                sorted_return = sorted_return[:self.cfg.top_k]
                max_return = sorted_return.max(0)[0]
                score = torch.exp(self.cfg.temperature * (sorted_return - max_return))
                cem_logits = torch.log((score[:, None, None, None] * encode_top_k_actions).sum(dim=0)) #(traj_length-seg_idx+1, action_dim, num_bins)
                cem_dist = D.categorical.Categorical(logits=cem_logits)
            
            top_k_states = decode["states"][sorted_indices[:self.cfg.top_k], -1]
            # print("policy_pred", policy_pred[0], "cem_logits", cem_logits[0])
            action_sample = action_values[cem_dist.sample()[0]]
            action_expert = action_values[torch.max(policy_pred, dim=-1)[1][0]]
            
            
        return action_sample, action_expert
    
    def compute_mtm_loss(self, batch: Dict[str, torch.Tensor]):
        
        #calculate future prediction loss
        losses = {}
        masked_losses = {}
        masked_c_losses = {}
        encoded_batch = self.tokenizer_manager.encode(batch)
        targets = encoded_batch
        torch_rcbc_mask = create_random_mask(self.cfg.traj_length, self.cfg.device, "rcbc")
        rcbc_preds = self.mtm(encoded_batch, torch_rcbc_mask)
        torch_fd_mask = create_random_mask(self.cfg.traj_length, self.cfg.device, "fd")
        fd_preds = self.mtm(encoded_batch, torch_fd_mask)
        for key in targets.keys():
            if key == "actions":
                target = targets[key]
                pred = rcbc_preds[key]
                mask = torch_rcbc_mask[key]
            else:
                target = targets[key]
                pred = fd_preds[key]
                mask = torch_fd_mask[key]
            
            if len(mask.shape) == 1:
                # only along time dimension: repeat across the given dimension
                mask = mask[:, None].repeat(1, target.shape[2])
            elif len(mask.shape) == 2:
                pass
            
            if key == "actions" :
                    raw_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
                ).unsqueeze(3)
            else: 
                raw_loss = nn.MSELoss(reduction="none")(pred, target)
            
            # raw_loss shape = [batch_size, T, P, 1]
            loss = raw_loss.sum(dim=(2, 3)).mean()
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
            loss = torch.sum(torch.stack([losses[key] * weight for key, weight in self.cfg.loss_weight.items()]))
        
        return loss, losses, masked_losses, masked_c_losses
    
    def compute_q_loss(self, batch):
        
        #convert batch data structure
        states = batch["states"][:, :-1]
        actions = batch["actions"][:, :-1]
        rewards = batch["rewards"][:, :-1]
        next_states = batch["states"][:, 1:]
        
        # Flatten the data for compatibility with the model
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = rewards.reshape(-1)
        next_states = next_states.reshape(-1, next_states.shape[-1])
    
        with torch.no_grad():
            next_v = self.value(next_states)
            target_q_values = rewards + self.cfg.discount * next_v
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = ((q1 - target_q_values)**2).mean() 
        critic2_loss = ((q2 - target_q_values)**2).mean()
        
        return critic1_loss, critic2_loss
    
    def compute_value_loss(self, batch):
        states = batch["states"][:, :-1]
        actions = batch["actions"][:, :-1]
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)   
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1,q2)
        
        value = self.value(states)
        value_loss = loss(min_Q - value, 0.8).mean()
        
        return value_loss
        
    
    def mtm_update(self, batch):
        loss, losses, masked_losses, masked_c_losses = self.compute_mtm_loss(batch)
        log_dict = {}
        for k, l in losses.items():
            log_dict[f"train/loss_{k}"] = l
            if k in masked_losses.keys():
                log_dict[f"train/masked_loss_{k}"] = masked_losses[k]
            if k in masked_c_losses.keys():
                log_dict[f"train/masked_c_loss_{k}"] = masked_c_losses[k]
        
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
        critic1_loss, critic2_loss = self.compute_q_loss(batch)
        
        log_dict = {}

        log_dict[f"train/q1_loss"] = critic1_loss.item()
        log_dict[f"train/q2_loss"] = critic2_loss.item()
        
        #backprop
        self.critic1.zero_grad(set_to_none=True)
        self.critic2.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic2_loss.backward()
        
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        return log_dict
    
    def value_update(self,
                     batch: Dict[str, torch.Tensor]):
        
        value_loss = self.compute_value_loss(batch)
        
        log_dict = {}
        
        log_dict[f"train/v_loss"] = value_loss
        
        self.value.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()
        
        return log_dict
        
        
    def critic_target_soft_update(self):
        
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data)
    
    
    def action_sample(self, sequence_history, percentage=1.0, p=[0,0.1, 0.2, 0.4, 0.2, 0.1, 0, 0]):
        
        
        end_idx = sequence_history["path_length"]
        if sequence_history["path_length"] < self.cfg.traj_length:
            p = [1, 0, 0, 0, 0, 0, 0, 0]
        #Seperate history and prediction
        sep_idx = np.random.choice(self.cfg.traj_length, p=p)

        _, obs_dim = sequence_history["observations"].shape
        action_dim = sequence_history["actions"].shape[-1]
        zero_trajectory = {
            "observations": np.zeros((1, self.cfg.traj_length, obs_dim)),
            "actions": np.zeros((1, self.cfg.traj_length, action_dim)),
            "rewards": np.zeros((1, self.cfg.traj_length, 1)),
            "values": np.zeros((1, self.cfg.traj_length, 1))
        }
        
        for k in zero_trajectory.keys():
            if sep_idx == 0:
                zero_trajectory[k][0, 0] = sequence_history[k][end_idx]
            else:
                zero_trajectory[k][0, :sep_idx + 1] = sequence_history[k][end_idx - sep_idx:end_idx + 1]
        
        torch_zero_trajectories = {
            "states" if k == "observations" else "returns" if k == "values" else k: torch.tensor(v, device=self.cfg.device) 
            for k, v in zero_trajectory.items()
        }
        
        return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
        return_min = self.tokenizer_manager.tokenizers["returns"].stats.min
        
        
        return_value = return_min + (return_max - return_min) * percentage
        return_to_go = float(return_value)
        returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
        torch_zero_trajectories["returns"] = torch.from_numpy(returns).to(self.cfg.device)
        
        sample, policy = self.compute_target_cem_action(torch_zero_trajectories, sep_idx)
        
        
        return sample, policy
        
    
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
            current_trajectory = {"observations": np.zeros((1000, self.env.observation_space.shape[0]), dtype=np.float32),
                                  "actions": np.zeros((1000, self.env.action_space.shape[0]), dtype=np.float32),
                                  "rewards": np.zeros((1000, 1), dtype=np.float32),
                                  "values": np.zeros((1000, 1), dtype=np.float32),
                                  "total_return": 0,
                                  "path_length": 0}
            
            observation, done = self.env.reset(), False
            if len(videos) < num_videos:
                try:
                    imgs = [self.env.sim.render(64, 48, camera_name="track")[::-1]]
                except:
                    imgs = [self.env.render()[::-1]]
            
            timestep = 0
            while not done and timestep < 1000:
                current_trajectory["observations"][timestep] = observation
                _, action = self.action_sample(current_trajectory, percentage=1.0, p=[0,0,0,0,0,0,0,1])
                action = np.clip(action.cpu().numpy(), -1, 1)
                new_observation, reward, done, info = self.env.step(action)
                current_trajectory["actions"][timestep] = action
                current_trajectory["rewards"][timestep] = reward
                observation = new_observation
                timestep += 1
                current_trajectory["path_length"] += 1
                if len(videos) < num_videos:
                    try:
                        imgs.append(self.env.sim.render(64, 48, camera_name="track")[::-1])
                    except:
                        imgs.append(self.env.render()[::-1])
            
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
        for idx, v in enumerate(videos):
            log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
                v.transpose(0, 3, 1, 2), fps=10, format="gif"
            )

        return log_data
    

def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)