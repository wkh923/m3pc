import numpy as np
import torch
from typing import List

from research.jaxrl.datasets.d4rl_dataset import D4RLDataset
from research.jaxrl.utils import make_env
from research.mtm.models.mtm_model import MTM
from research.mtm.datasets.base import DataStatistics
from research.mtm.datasets.sequence_dataset import Trajectory, segment
from research.mtm.tokenizers.base import TokenizerManager
from research.finetune.masks import create_rcbc_mask

class ReplayBuffer:
    """For trajectory transformer
    """
    def __init__(
        self,
        dataset: D4RLDataset,
        discount: float = 0.99,
        sequence_length: int = 32,
        max_path_length: int = 1000,
        batch_size: int = 256,
        buffer_size: int = 1000,
        use_reward: bool = True,
        name: str = "",
        mode: str = "random",
        max_iterations: int = 10
    ):
        self.env = dataset.env
        self.dataset = dataset
        self.max_path_length = max_path_length
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._use_reward = use_reward
        self._name = name
        self._mode = mode 
        self.max_iterations = max_iterations

        # extract data from Dataset
        self.observations_raw = dataset.observations
        self.actions_raw = dataset.actions
        self.rewards_raw = dataset.rewards.reshape(-1, 1)
        self.terminals_raw = dataset.dones_float

        ## segment
        self.actions_segmented, self.termination_flags, self.path_lengths = segment(
            self.actions_raw, self.terminals_raw, max_path_length
        )
        self.observations_segmented, *_ = segment(
            self.observations_raw, self.terminals_raw, max_path_length
        )
        self.rewards_segmented, *_ = segment(
            self.rewards_raw, self.terminals_raw, max_path_length
        )

        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False

        self.discounts = (self.discount ** np.arange(self.max_path_length))[:, None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:, t + 1 :] * self.discounts[: -t - 1]).sum(
                axis=1
            )
            self.values_segmented[:, t] = V
        
        N_p, Max_Path_Len, _ = self.values_segmented.shape
        if self.use_avg:
            divisor = np.arange(1, Max_Path_Len + 1)[::-1][None, :, None]
            self.values_segmented = self.values_segmented / divisor

        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]

        self.observation_dim = self.observations_raw.shape[1]
        self.action_dim = self.actions_raw.shape[1]

        assert (
            self.observations_segmented.shape[0]
            == self.actions_segmented.shape[0]
            == self.rewards_segmented.shape[0]
            == self.values_segmented.shape[0]
        )
        #  assert len(set(self.path_lengths)) == 1
        
        self.trajectory_returns = self.rewards_segmented[:, :].sum(axis=(1, 2), keepdims=False) ##[n_paths]
        sorted_index = np.argsort(self.trajectory_returns, axis = 0)[::-1]
        self.path_lengths = np.array(self.path_lengths)[sorted_index]
        self.observations_segmented = self.observations_segmented[sorted_index]
        self.actions_segmented = self.actions_segmented[sorted_index]
        self.rewards_segmented = self.rewards_segmented[sorted_index]
        self.values_segmented = self.values_segmented[sorted_index]
        self.trajectory_returns = self.trajectory_returns[sorted_index]
        
        keep_idx = []
        traj_count = 0
        for idx, pl in enumerate(self.path_lengths):
            if traj_count == self.buffer_size:
                break
            elif pl < sequence_length: 
                pass
            else:
                keep_idx.append(idx)
                traj_count += 1
        
        self.path_lengths = self.path_lengths[keep_idx]
        self.observations_segmented = self.observations_segmented[keep_idx]
        self.actions_segmented = self.actions_segmented[keep_idx]
        self.rewards_segmented = self.rewards_segmented[keep_idx]
        self.values_segmented = self.values_segmented[keep_idx]
        self.trajectory_returns = self.trajectory_returns[keep_idx]
        self.p = self.trajectory_returns / self.trajectory_returns.sum(axis=0)
        
    def online_rollout(self,
                       model: MTM,
                       tokenizer_manager: TokenizerManager,
                       device: str,
                       num_trajectories: int = 300,
                       percentage: float = 1.0,
                       clip_min: float = -0.1,
                       clip_max: float = 0.1,
                       action_noise_std: float = 0.2
                       ):
        
        assert num_trajectories <= self.buffer_size
        model.eval()
        return_max = tokenizer_manager.tokenizers["returns"].stats.max
        return_min = tokenizer_manager.tokenizers["returns"].stats.min
        
        return_value = return_min + (return_max - return_min) * percentage
        return_to_go = float(return_value)
        
        zero_trajectory = {
            "states": np.zeros(self.sequence_length, self.observation_dim),
            "actions": np.zeros(self.sequence_length, self.action_dim),
            "rewards": np.zeros(self.sequence_length, 1),
            "returns": np.zeros(self.sequence_length, 1)
        }
        torch_rcbc_mask = create_rcbc_mask(self.sequence_length, device)
        new_trajectories = []
        
        for _ in range(num_trajectories):
            
            current_trajectory = {"observations": np.zeros(self.max_path_length, self.observation_dim),
                                  "actions": np.zeros(self.max_path_length, self.action_dim),
                                  "rewards": np.zeros(self.max_path_length, 1),
                                  "values": np.zeros(self.max_path_length, 1),
                                  "total_return": 0,
                                  "path_length": 0}
            torch_trajectory = {
            k: torch.tensor(v, device=device)[None] for k, v in zero_trajectory.items()}
            
            observation, done = self.env.reset(), False
            timestep = 0
            while not done and timestep < self.max_path_length:
                torch_trajectory["returns"][0, 0] = torch.tensor([return_to_go])
                torch_trajectory["states"][0, 0] = torch.tensor(observation)
                encode = tokenizer_manager.encode(torch_trajectory)
                with torch.no_grad():
                    pred = model(encode, torch_rcbc_mask)
                action = tokenizer_manager.decode(pred)["action"][0, 0].cpu().numpy()
                action = np.clip(action + action_noise_std * np.random.randn(self.action_dim), clip_min, clip_max)
                new_observation, reward, done, _ = self.env.step(action)
                current_trajectory["observations"][timestep] = observation
                current_trajectory["actions"][timestep] = action
                current_trajectory["rewards"][timestep] = reward
                observation = new_observation
                timestep += 1
            
            for t in range(self.max_path_length):
                V = (current_trajectory["rewards"][t + 1 :] * self.discounts[: -t - 1]).sum(
                    axis=0
                )
                current_trajectory["values"][t] = V
            
            if self.use_avg:
                divisor = np.arange(1, self.max_path_length + 1)[::-1][:, None]
                current_trajectory["values"] = current_trajectory["values"] / divisor
            
            current_trajectory["total_return"] = current_trajectory["rewards"].sum()
            current_trajectory["path_length"] = timestep
            
            new_trajectories.append(current_trajectory)
            self.update_buffer(new_trajectories)
    
    
    
    def update_buffer(self,
                      new_trajectories: List):
        
        num_new_trajectories = len(new_trajectories)
        
        new_path_lengths = np.array([traj["path_length"] for traj in new_trajectories])
        new_trajectory_returns = np.array([traj["total_return"] for traj in new_trajectories])
        new_observations = np.stack([traj["observations"] for traj in new_trajectories], axis=0)
        new_actions = np.stack([traj["actions"] for traj in new_trajectories], axis=0)
        new_values = np.stack([traj["values"] for traj in new_trajectories], axis=0)
        new_rewards = np.stack([traj["rewards"] for traj in new_trajectories], axis=0)
        
        
        self.path_lengths = np.concatenate((self.path_lengths[num_new_trajectories:], new_path_lengths), axis=0)
        self.observations_segmented = np.concatenate((self.observations_segmented[num_new_trajectories:], new_observations), axis=0)
        self.actions_segmented = np.concatenate((self.actions_segmented[num_new_trajectories:], new_actions), axis=0)
        self.rewards_segmented = np.concatenate((self.rewards_segmented[num_new_trajectories:], new_rewards), axis=0)
        self.values_segmented = np.concatenate((self.values_segmented[num_new_trajectories:], new_values), axis=0)
        self.trajectory_returns = np.concatenate((self.trajectory_returns[num_new_trajectories:], new_trajectory_returns), axis=0)
        self.p = self.trajectory_returns / self.trajectory_returns.sum(axis=0)
    
    def sample(self):
        """Sample a batch of trajectories from the replay buffer.
        
        Depending on the mode, the sampling will be uniform or based on trajectory return probabilities.
        
        Returns:
            Dict containing observations, actions, rewards, and values for the sampled subsequences.
        """
        if self._mode == "uniform":
            # Sample uniformly without considering trajectory return probabilities
            sampled_indices = np.random.choice(len(self.observations_segmented), size=self.batch_size, replace=False)

        elif self._mode == "prob":
            # Sample indices based on trajectory return probabilities
            sampled_indices = np.random.choice(np.arange(len(self.p)), size=self.batch_size, p=self.p)
        else:
            raise ValueError(f"Invalid sampling mode: {self._mode}")

        # Extract subsequences for the sampled indices
        sampled_observations, sampled_actions, sampled_rewards, sampled_values = [], [], [], []

        for index in sampled_indices:
            start_idx = np.random.randint(0, self.path_lengths[index] - self.sequence_length + 1) 
            end_idx = start_idx + self.sequence_length

            sampled_observations.append(self.observations_segmented[index, start_idx:end_idx])
            sampled_actions.append(self.actions_segmented[index, start_idx:end_idx])
            sampled_rewards.append(self.rewards_segmented[index, start_idx:end_idx])
            sampled_values.append(self.values_segmented[index, start_idx:end_idx])

        batch = {
            'states': np.stack(sampled_observations),
            'actions': np.stack(sampled_actions),
            'rewards': np.stack(sampled_rewards),
            'returns': np.stack(sampled_values)
        }
        
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch
    
    def __iter__(self):
        """Initializes the iterator and returns the object itself."""
        # Initializing current index to 0. This will be used to keep track of batches yielded.
        self.current_index = 0
        return self

    def __next__(self):
        """Returns the next batch. If all batches are already fetched, raises StopIteration."""
        # If we've already yielded the maximum number of batches, stop the iteration.
        if self.current_index >= self.max_iterations:
            raise StopIteration
        self.current_index += 1
        # Return the next batch.
        return self.sample()
            
            
            
    
    
        
            
            
                
                
                
                
                
            
            

        