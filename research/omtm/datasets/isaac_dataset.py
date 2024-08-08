import research.omtm.datasets.sequence_dataset as seq_d
import os
import json
import h5py
import numpy as np

import h5py


def segment(observations, terminals, max_path_length):
    """
    segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros(
        (n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype
    )
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i, :path_length] = traj
        early_termination[i, path_length:] = 1

    return trajectories_pad, early_termination, path_lengths


class ISAACDataset(seq_d.SequenceDataset):

    def __init__(
        self,
        dataset_dir: str = "/workspace/m3pc/research/trajectories.h5",
        discount: float = 0.99,
        sequence_length: int = 8,
        max_path_length: int = 1000,
        use_reward: bool = True,
        name: str = "",
    ):
        # self.env = dataset.env  # TODO: env for dataset
        # load dataset from h5 file
        # <KeysViewHDF5 ['actions', 'dones', 'next_states', 'rewards', 'states']>

        with h5py.File(dataset_dir, "r") as f:
            # first load the h5 file to a np cache
            observations_np = np.array(f["states"][:])
            actions_np = np.array(f["actions"][:])
            rewards_np = np.array(f["rewards"][:])
            terminals_np = np.array(f["dones"][:])

        # find The per-dim mean and zoom-factor to make states and actions have zero mean and stay strictly within [-1+e-4, 1+e-4]
        # (length, dim)
        self.observation_mean = np.mean(observations_np, axis=0)
        self.observation_range = np.max(
            np.abs(observations_np - self.observation_mean), axis=0
        )
        self.action_mean = np.mean(actions_np, axis=0)
        self.action_range = np.max(np.abs(actions_np - self.action_mean), axis=0)

        # print("observations_np", observations_np[0])

        # for range element is too small, set it to 1e2
        self.observation_range[self.observation_range < 1e-2] = 1e-2

        # normalize the states and actions
        observations_np = (observations_np - self.observation_mean) / (
            self.observation_range
        )

        # print("observation_range", self.observation_range)
        actions_np = (actions_np - self.action_mean) / (self.action_range)

        # print(actions_np)

        # further clip the states and actions to [-1+e-4, 1+e-4]
        observations_np = np.clip(observations_np, -1 + 1e-4, 1 - 1e-4)
        actions_np = np.clip(actions_np, -1 + 1e-4, 1 - 1e-4)

        # further normalize the rewards, but don't clip them, using normal mean and std
        self.reward_mean = np.mean(rewards_np)
        self.reward_std = np.std(rewards_np)
        rewards_np = (rewards_np - self.reward_mean) / self.reward_std

        self.norm_dict = {
            "obs_mean": self.observation_mean,
            "obs_std": self.observation_range,
            "act_mean": self.action_mean,
            "act_std": self.action_range,
            "rew_mean": self.reward_mean,
            "rew_std": self.reward_std
        }

        # then load the np cache to the dataset
        self.observations_raw = observations_np
        self.actions_raw = actions_np
        self.rewards_raw = rewards_np
        self.terminals_raw = terminals_np

        self.max_path_length = max_path_length
        self.sequence_length = sequence_length
        self._use_reward = use_reward
        self._name = name

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
        keep_idx = []
        index_map = {}
        count = 0
        traj_count = 0
        for idx, pl in enumerate(self.path_lengths):
            if pl < sequence_length:
                pass
            else:
                keep_idx.append(idx)
                for i in range(pl - sequence_length + 1):
                    index_map[count] = (traj_count, i)
                    count += 1
                traj_count += 1

        self.index_map = index_map
        self.path_lengths = np.array(self.path_lengths)[keep_idx]
        self.observations_segmented = self.observations_segmented[keep_idx]
        self.actions_segmented = self.actions_segmented[keep_idx]
        self.rewards_segmented = self.rewards_segmented[keep_idx]
        self.values_segmented = self.values_segmented[keep_idx]
        self.num_trajectories = self.observations_segmented.shape[0]

        self.raw_data = {
            "states": self.observations_raw,
            "actions": self.actions_raw,
            "rewards": self.rewards_raw,
            "returns": self.values_raw,
        }


if __name__ == "__main__":
    dataset = ISAACDataset()
    print(dataset.observations_segmented.shape)
    print(dataset.actions_segmented.shape)
    print(dataset.rewards_segmented.shape)
    print(dataset.values_segmented.shape)