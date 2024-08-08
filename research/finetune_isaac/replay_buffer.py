import numpy as np
import torch
import random
from typing import List, Callable
from collections import deque, namedtuple

from research.jaxrl.datasets.d4rl_dataset import D4RLDataset
from research.jaxrl.utils import make_env
from research.omtm.datasets.sequence_dataset import Trajectory, segment
from research.finetune_omtm.masks import *


class ReplayBuffer:
    """For trajectory transformer"""

    def __init__(
        self,
        cfg,
        dataset: D4RLDataset,
        discount: float = 0.99,
        max_path_length: int = 1000,
        use_reward: bool = True,
        shuffle: bool = True,
    ):
        self.cfg = cfg
        self.env = dataset.env
        self.dataset = dataset
        self.max_path_length = max_path_length
        self.sequence_length = cfg.traj_length
        self.traj_batch_size = cfg.traj_batch_size
        self.traj_buffer_size = cfg.traj_buffer_size
        self.trans_batch_size = cfg.trans_batch_size
        self.trans_buffer_size = cfg.trans_buffer_size
        self.buffer_init_size = int(cfg.buffer_init_ratio * self.trans_buffer_size)
        self._use_reward = use_reward
        self._name = cfg.env_name
        self._mode = cfg.select_mode
        self.mtm_iter = cfg.mtm_iter_per_rollout
        self.v_iter = cfg.v_iter_per_mtm
        self.total_step = 0

        # extract data from Dataset
        self.observations_raw = dataset.observations
        self.obs_mean = self.observations_raw.mean(axis=0)
        self.obs_std = self.observations_raw.std(axis=0)
        self.actions_raw = dataset.actions
        self.rewards_raw = dataset.rewards.reshape(-1, 1)
        self.dones_raw = dataset.dones_float
        self.terminals_raw = dataset.terminals_float
        self.next_observations_raw = dataset.next_observations

        ## segment
        self.actions_segmented, self.termination_flags, self.path_lengths = segment(
            self.actions_raw, self.dones_raw, max_path_length
        )
        self.observations_segmented, *_ = segment(
            self.observations_raw, self.dones_raw, max_path_length
        )
        self.rewards_segmented, *_ = segment(
            self.rewards_raw, self.dones_raw, max_path_length
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

        self.offline_trans_buffer = deque(maxlen=self.trans_buffer_size)
        self.online_trans_buffer = deque(maxlen=self.trans_buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        
        
        sorted_idx_raw = np.argsort(self.rewards_raw[:, 0], axis=0)[::-1][
            : self.buffer_init_size
        ]
        np.random.shuffle(sorted_idx_raw)
        self.sorted_observations_raw = self.observations_raw[sorted_idx_raw]
        self.sorted_actions_raw = self.actions_raw[sorted_idx_raw]
        self.sorted_rewards_raw = self.rewards_raw[sorted_idx_raw]
        self.sorted_terminals_raw = self.terminals_raw[sorted_idx_raw]
        self.sorted_next_observations_raw = self.next_observations_raw[
            sorted_idx_raw
        ]
        for state, action, reward, next_state, done in zip(
            self.sorted_observations_raw,
            self.sorted_actions_raw,
            self.sorted_rewards_raw,
            self.sorted_next_observations_raw,
            self.sorted_terminals_raw,
        ):
            e = self.experience(state, action, reward, next_state, done)
            self.offline_trans_buffer.append(e)


        #  assert len(set(self.path_lengths)) == 1

        self.trajectory_returns = self.rewards_segmented[:, :].sum(
            axis=(1, 2), keepdims=False
        )  ##[n_paths]
        sorted_index = np.argsort(self.trajectory_returns, axis=0)[::-1]
        self.path_lengths = np.array(self.path_lengths)[sorted_index]
        self.observations_segmented = self.observations_segmented[sorted_index]
        self.actions_segmented = self.actions_segmented[sorted_index]
        self.rewards_segmented = self.rewards_segmented[sorted_index]
        self.values_segmented = self.values_segmented[sorted_index]
        self.trajectory_returns = self.trajectory_returns[sorted_index]

        keep_idx = []
        traj_count = 0
        for idx, pl in enumerate(self.path_lengths):
            if traj_count == self.traj_buffer_size:
                break
            elif pl < self.sequence_length:
                pass
            else:
                keep_idx.append(idx)
                traj_count += 1

        if shuffle:
            shuffle_indices = np.arange(len(keep_idx))
            np.random.shuffle(shuffle_indices)
            keep_idx = [keep_idx[i] for i in shuffle_indices]
        self.path_lengths = self.path_lengths[keep_idx]
        self.path_lengths_avg = np.mean(self.path_lengths)
        self.observations_segmented = self.observations_segmented[keep_idx]
        self.actions_segmented = self.actions_segmented[keep_idx]
        self.rewards_segmented = self.rewards_segmented[keep_idx]
        self.values_segmented = self.values_segmented[keep_idx]
        self.values_up_bound = self.values_segmented.max(axis=0)
        self.trajectory_returns = self.trajectory_returns[keep_idx]
        self.p = self.path_lengths / self.path_lengths.sum(axis=0)
        self.p_length_list = []
        self.p_return_list = []


    def online_rollout(
        self,
        sample_func: Callable,
        num_trajectories: int = 1,
    ) -> Dict:
        """Collect trajectories from the environment using the provided policy.


        Args:
            sample_func: A function that samples actions from the policy.
            num_trajectories: Number of trajectories to collect.


        Returns:
            A dictionary containing the mean return and mean path length of the collected trajectories.
        """

        assert num_trajectories <= self.traj_buffer_size
        assert (
            num_trajectories == 1
        ), "Only support one trajectory collection! for more than one, equal to lower the update frequency for each online rollout."
        new_trajectories = []

        for _ in range(num_trajectories):

            current_trajectory = {
                "observations": np.zeros(
                    (self.max_path_length, self.observation_dim), dtype=np.float32
                ),
                "actions": np.zeros(
                    (self.max_path_length, self.action_dim), dtype=np.float32
                ),
                "rewards": np.zeros((self.max_path_length, 1), dtype=np.float32),
                "values": np.zeros((self.max_path_length, 1), dtype=np.float32),
                "total_return": 0,
                "path_length": 0,
            }

            observation, done = self.env.reset(), False
            timestep = 0
            while not done and timestep < self.max_path_length:
                current_trajectory["observations"][timestep] = observation
                action = sample_func(
                    current_trajectory,
                    percentage=self.cfg.rtg_percent,
                    plan=self.cfg.plan,
                )
                action = np.clip(
                    action.cpu().numpy(), self.cfg.clip_min, self.cfg.clip_max
                )
                new_observation, reward, done, _ = self.env.step(action)
                real_done = False  # Episode can timeout which is different from done
                if done and timestep < self.max_path_length - 1:
                    real_done = True
                    print("Episode terminated early.")
                elif timestep == self.max_path_length - 1:
                    print("1000! not done yet!")
                e = self.experience(
                    observation, action, reward, new_observation, real_done
                )  # done is always True at 1000th step
                
                self.online_trans_buffer.append(e)
                current_trajectory["actions"][timestep] = action
                current_trajectory["rewards"][timestep] = reward
                observation = new_observation
                timestep += 1
                current_trajectory["path_length"] += 1

            for t in range(self.max_path_length):
                V = (
                    current_trajectory["rewards"][t + 1 :] * self.discounts[: -t - 1]
                ).sum(axis=0)
                current_trajectory["values"][t] = V

            if self.use_avg:
                divisor = np.arange(1, self.max_path_length + 1)[::-1][:, None]
                current_trajectory["values"] = current_trajectory["values"] / divisor

            current_trajectory["total_return"] = current_trajectory["rewards"].sum()
            self.p_length_list.append(current_trajectory["path_length"])
            self.total_step += current_trajectory["path_length"]
            self.p_return_list.append(current_trajectory["total_return"])
            new_trajectories.append(current_trajectory)

        if len(new_trajectories) > 0:
            self.update_buffer(new_trajectories)
            explore_step_mean = np.mean(
                np.array([traj["path_length"] for traj in new_trajectories])
            )
            explore_return_mean = np.mean(
                np.array([traj["total_return"] for traj in new_trajectories])
            )
            log_dict = {
                "explore/rollout_steps_mean": explore_step_mean,
                "explore/rollout_return_mean": explore_return_mean,
            }

        # TODO: assume only one trajectory is collected
        else:
            log_dict = {
                "explore/rollout_steps_mean": current_trajectory["path_length"],
                "explore/rollout_return_mean": current_trajectory["total_return"],
            }
            print("No new trajectories collected, because they are too short.")
        return log_dict

    def update_buffer(self, new_trajectories: List):

        num_new_trajectories = len(new_trajectories)

        new_path_lengths = np.array([traj["path_length"] for traj in new_trajectories])
        print("new_path_lengths", new_path_lengths)
        new_trajectory_returns = np.array(
            [traj["total_return"] for traj in new_trajectories]
        )
        print("new_trajectory_returns", new_trajectory_returns)
        new_observations = np.stack(
            [traj["observations"] for traj in new_trajectories], axis=0
        )
        new_actions = np.stack([traj["actions"] for traj in new_trajectories], axis=0)
        new_values = np.stack([traj["values"] for traj in new_trajectories], axis=0)
        new_rewards = np.stack([traj["rewards"] for traj in new_trajectories], axis=0)

        self.path_lengths = np.concatenate(
            (self.path_lengths[num_new_trajectories:], new_path_lengths), axis=0
        )
        self.path_lengths_avg = np.mean(self.path_lengths)
        self.observations_segmented = np.concatenate(
            (self.observations_segmented[num_new_trajectories:], new_observations),
            axis=0,
        )
        self.actions_segmented = np.concatenate(
            (self.actions_segmented[num_new_trajectories:], new_actions), axis=0
        )
        self.rewards_segmented = np.concatenate(
            (self.rewards_segmented[num_new_trajectories:], new_rewards), axis=0
        )
        self.values_segmented = np.concatenate(
            (self.values_segmented[num_new_trajectories:], new_values), axis=0
        )
        self.trajectory_returns = np.concatenate(
            (self.trajectory_returns[num_new_trajectories:], new_trajectory_returns),
            axis=0,
        )
        self.p = self.path_lengths / self.path_lengths.sum(axis=0)
        self.values_up_bound = self.values_segmented.max(axis=0)

    def traj_sample(self):
        """Sample a batch of trajectories from the replay buffer.

        Depending on the mode, the sampling will be uniform or based on trajectory return probabilities.

        Returns:
            Dict containing observations, actions, rewards, and values for the sampled subsequences.
        """
        if self._mode == "uniform":
            # Sample uniformly without considering trajectory return probabilities
            sampled_indices = np.random.choice(
                len(self.observations_segmented),
                size=self.traj_batch_size,
                replace=True,
            )

        elif self._mode == "prob":
            # Sample indices based on trajectory return probabilities
            sampled_indices = np.random.choice(
                np.arange(len(self.p)), size=self.traj_batch_size, p=self.p
            )
        else:
            raise ValueError(f"Invalid sampling mode: {self._mode}")

        # Extract subsequences for the sampled indices
        sampled_observations, sampled_actions, sampled_rewards, sampled_values = (
            [],
            [],
            [],
            [],
        )

        for index in sampled_indices:
            start_idx = np.random.randint(
                0, self.path_lengths[index] - self.sequence_length + 1
            )
            end_idx = start_idx + self.sequence_length

            sampled_observations.append(
                self.observations_segmented[index, start_idx:end_idx]
            )
            sampled_actions.append(self.actions_segmented[index, start_idx:end_idx])
            sampled_rewards.append(self.rewards_segmented[index, start_idx:end_idx])
            sampled_values.append(self.values_segmented[index, start_idx:end_idx])

        batch = {
            "states": np.stack(sampled_observations),
            "actions": np.stack(sampled_actions),
            "rewards": np.stack(sampled_rewards),
            "returns": np.stack(sampled_values),
        }

        batch = {k: torch.from_numpy(v).to(self.cfg.device) for k, v in batch.items()}

        return batch

    def trans_sample(self):
        """ "Sample a batch of experinces from transition level replay buffer"""
        if len(self.online_trans_buffer) < self.cfg.using_online_threshold:
            experiences = random.sample(self.offline_trans_buffer, k=self.trans_batch_size)
        else:
            online_experiences = random.sample(self.online_trans_buffer, k=self.trans_batch_size//2)
            offline_experiences = random.sample(self.offline_trans_buffer, k=(self.trans_batch_size - self.trans_batch_size//2))
            experiences = online_experiences + offline_experiences

        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.cfg.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.cfg.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.cfg.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.cfg.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.cfg.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __iter__(self):
        """Initializes the iterator and returns the object itself."""
        # Initializing current index to 0. This will be used to keep track of batches yielded.
        self.current_index = 0
        return self

    def __next__(self):
        """Returns the next batch. If all batches are already fetched, raises StopIteration."""
        # If we've already yielded the maximum number of batches, stop the iteration.
        if self.current_index >= self.mtm_iter:
            raise StopIteration
        self.current_index += 1
        # Return the next batch.
        return self.traj_sample()
