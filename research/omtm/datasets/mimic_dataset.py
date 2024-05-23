from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Sequence

import tqdm
import research.omtm.datasets.sequence_dataset as seq_d
from research.omtm.datasets.sequence_dataset import segment
import os
import numpy as np

import robomimic.utils.file_utils as FileUtils
import numpy as np
import gym as gym
from gym import spaces, Env

from robosuite.wrappers import Wrapper
import wandb


from torch.utils.data import DataLoader
import torch

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.env_utils as EnvUtils

from robomimic.config import config_factory
from robomimic.algo import algo_factory

# import robosuite as suite

# the dataset registry can be found at robomimic/__init__.py
from robomimic import DATASET_REGISTRY

from research.omtm.tokenizers.base import TokenizerManager


@dataclass(frozen=True)
class Trajectory:
    """Immutable container for a Trajectory.

    Each has shape (T, X).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    def __post_init__(self):
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.observations.shape[0] == self.rewards.shape[0]

    def __len__(self) -> int:
        return self.observations.shape[0]

    @staticmethod
    def create_empty(
        observation_shape: Sequence[int], action_shape: Sequence[int]
    ) -> "Trajectory":
        """Create an empty trajectory."""
        return Trajectory(
            observations=np.zeros((0,) + observation_shape),
            actions=np.zeros((0,) + action_shape),
            rewards=np.zeros((0, 1)),
        )

    def append(
        self, observation: np.ndarray, action: np.ndarray, reward: float
    ) -> "Trajectory":
        """Append a new observation, action, and reward to the trajectory."""
        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        return Trajectory(
            observations=np.concatenate((self.observations, observation[None])),
            actions=np.concatenate((self.actions, action[None])),
            rewards=np.concatenate((self.rewards, np.array([reward])[None])),
        )


@torch.inference_mode()
def sample_action_bc(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))
    returns = np.zeros((traj_len, 1))

    # print("observation_shape", observation_shape)
    # print("observation", observation)

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"].mean[0][0][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc2(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage=1.0,
):
    traj_len = model.max_len
    # make observations and actions

    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))

    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min

    return_value = return_min + (return_max - return_min) * percentage
    return_to_go = float(return_value)
    returns = return_to_go * np.ones((traj_len, 1))

    masks = np.zeros(traj_len)

    i = -1
    max_len = min(traj_len - 1, len(traj))
    assert max_len < traj_len
    for i in range(max_len):
        observations[i] = traj.observations[-max_len + i]
        actions[i] = traj.actions[-max_len + i]
        # rewards[i] = traj.rewards[-max_len + i]
        masks[i] = 1

    assert i == max_len - 1
    # fill in the rest with the current observation
    observations[i + 1] = observation
    obs_mask = np.copy(masks)
    obs_mask[i + 1] = 1

    # pass through tokenizer
    trajectories = {
        "states": observations,
        "actions": actions,
        "returns": returns,
    }

    reward_mask = np.ones(traj_len)
    masks = {
        "states": obs_mask,
        "actions": masks,
        "returns": reward_mask,
    }

    # convert to tensors and add
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}

    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"].mean[0][i + 1][0].cpu().numpy()
    return a


@torch.inference_mode()
def sample_action_bc_two_stage(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage=1.0,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))

    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min

    return_value = return_min + (return_max - return_min) * percentage
    return_to_go = float(return_value)
    returns = return_to_go * np.ones((traj_len, 1))

    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    ret_mask = np.zeros(traj_len)
    ret_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        "returns": ret_mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # fill in predicted states
    torch_trajectories["states"] = torch_trajectories["states"] * torch_masks["states"][
        None, :, None
    ] + decode["states"] * (1 - torch_masks["states"][None, :, None])
    # fill in predicted returns
    torch_trajectories["returns"] = torch_trajectories["returns"] * torch_masks[
        "returns"
    ][None, :, None] + decode["returns"] * (1 - torch_masks["returns"][None, :, None])

    # update masks
    masks = {
        "states": np.ones(traj_len),
        "actions": mask,
        "rewards": mask,
        "returns": np.ones(traj_len),
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"].mean[0][0][0].cpu().numpy()
    return a


SampleActions = Callable[[np.ndarray, Trajectory], np.ndarray]
"""Sample an action given the current observation and past history.

Parameters
----------
observation : np.ndarray, shape=(O,)
    Observation at time t.
trajectory_history : Trajectory
    History of observations, actions, and rewards from 0 to t-1.

Returns
-------
jnp.ndarray, shape=(A,)
    The sampled action.
"""


class MIMICDataset(seq_d.SequenceDataset):

    def __init__(
        self,
        dataset_path: str = "/home/hu/robomimic/datasets/can/paired/low_dim_v141.hdf5",
        discount: float = 0.99,
        sequence_length: int = 32,
        max_path_length: int = 300,
        use_reward: bool = True,
        name: str = "",
        max_eval_steps: int = 1000,
    ):
        self.max_eval_steps = max_eval_steps
        assert os.path.exists(
            dataset_path
        ), f"Dataset path {dataset_path} does not exist"

        # load dataset and transform it into the format is raw
        file_dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=(  # observations we want to appear in batches
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ),
            dataset_keys=(  # can optionally specify more keys here if they should appear in batches
                "actions",
                "rewards",
                "dones",
            ),
            load_next_obs=False,
            frame_stack=1,
            seq_length=1,  # length-10 temporal sequences
            pad_frame_stack=True,
            pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,  # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,  # can optionally provide a filter key here
        )
        print("\n============= Created Dataset =============")

        focus_keys = ["actions", "rewards", "dones", "obs"]  # ep is each length

        list_dict = {key: [] for key in focus_keys}
        obs_keys = file_dataset.obs_keys

        for d in range(len(file_dataset.demos)):
            traj_dict = file_dataset.get_trajectory_at_index(d)
            # traj_dict is a dict with keys and array as values
            # concate all the obs_keys in traj_dict["obs"] as a numpy array

            np_obs = np.concatenate([traj_dict["obs"][key] for key in obs_keys], axis=1)

            # print dim for each key
            if d == 0:
                print("griper_qpos", traj_dict["obs"]["robot0_gripper_qpos"])
                for key in obs_keys:
                    print(key, traj_dict["obs"][key].shape)

            np_actions = np.array(traj_dict["actions"])
            np_rewards = np.array(traj_dict["rewards"])
            np_dones = np.array(traj_dict["dones"])

            # print(
            #     "all the shapes",
            #     np_obs.shape,
            #     np_actions.shape,
            #     np_rewards.shape,
            #     np_dones.shape,
            # )

            # find the index of the first "1" in np_dones

            done_index = np.argmax(np_dones > 0)
            if d % 33 == 0:
                print(np_dones)
                print(
                    "cuted last part of the trajectory: ",
                    np_dones.shape[0] - done_index,
                )
            # concatenate the nps in corrisponding dict keys
            list_dict["obs"].append(np_obs[: done_index + 1])
            list_dict["actions"].append(np_actions[: done_index + 1])
            list_dict["rewards"].append(np_rewards[: done_index + 1])
            list_dict["dones"].append(np_dones[: done_index + 1])

        np_dict = {keys: np.concatenate(list_dict[keys], axis=0) for keys in focus_keys}

        # extract data from Dataset
        self.observations_raw = np_dict["obs"]
        self.actions_raw = np_dict["actions"]
        self.rewards_raw = np_dict["rewards"].reshape(-1, 1)
        self.terminals_raw = np_dict["dones"].reshape(-1, 1)

        # clip actions_raw to [-1,1]
        self.actions_raw = np.clip(self.actions_raw, -1 + 1e-4, 1 - 1e-4)

        # make env based on dataset meta and wrap it with gym wrapper
        dummy_spec = dict(
            obs=dict(
                low_dim=[
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object",
                ],
                rgb=[],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=True,
            render_offscreen=False,
            use_image_obs=False,
        )

        # self.env = env

        self.env = OldGymWrapper(
            env,
            keys=[
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ],
        )

        # self.dataset = dataset
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
            # self.use_avg = True
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

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        log_data = {}
        observation_shape = self.observations_raw.shape
        action_shape = self.actions_raw.shape
        device = next(model.parameters()).device

        bc_sampler = lambda o, t: sample_action_bc(
            o, t, model, tokenizer_manager, observation_shape, action_shape, device
        )
        results, videos = evaluate(
            bc_sampler,
            self.env,
            10,  # TODO: for test 1
            (self.observation_dim,),
            (self.action_dim,),
            num_videos=0,
            max_steps=self.max_eval_steps,
        )
        for k, v in results.items():
            log_data[f"eval_bc/{k}"] = v
        for idx, v in enumerate(videos):
            log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
                v.transpose(0, 3, 1, 2), fps=10, format="gif"
            )

        if "returns" in tokenizer_manager.tokenizers:
            for p in [1.0]:
                bc_sampler = lambda o, t: sample_action_bc2(
                    o,
                    t,
                    model,
                    tokenizer_manager,
                    observation_shape,
                    action_shape,
                    device,
                    percentage=p,
                )
                results, videos = evaluate(
                    bc_sampler,
                    self.env,
                    10,
                    (self.observation_dim,),
                    (self.action_dim,),
                    num_videos=0,
                    max_steps=self.max_eval_steps,
                )
                for k, v in results.items():
                    log_data[f"eval2/p={p}_{k}"] = v
                for idx, v in enumerate(videos):
                    log_data[f"eval2_video_{idx}/p={p}_video"] = wandb.Video(
                        v.transpose(0, 3, 1, 2), fps=10, format="gif"
                    )

        # if "returns" in tokenizer_manager.tokenizers:
        #     for p in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
        #         bc_sampler = lambda o, t: sample_action_bc_two_stage(
        #             o,
        #             t,
        #             model,
        #             tokenizer_manager,
        #             observation_shape,
        #             action_shape,
        #             device,
        #             percentage=p,
        #         )
        #         results, videos = evaluate(
        #             bc_sampler,
        #             self.dataset.env,
        #             20,
        #             (self.observation_dim,),
        #             (self.action_dim,),
        #             num_videos=0,
        #         )
        #         for k, v in results.items():
        #             log_data[f"eval_ts/p={p}_{k}"] = v
        #         for idx, v in enumerate(videos):
        #             log_data[f"eval_ts_video_{idx}/p={p}_video"] = wandb.Video(
        #                 v.transpose(0, 3, 1, 2), fps=10, format="gif"
        #             )

        return log_data


def evaluate(
    sample_actions: SampleActions,
    env: gym.Env,
    num_episodes: int,
    observation_space: Tuple[int, ...],
    action_space: Tuple[int, ...],
    disable_tqdm: bool = True,
    verbose: bool = False,
    all_results: bool = False,
    num_videos: int = 3,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    # stats: Dict[str, Any] = {"return": [], "length": []}
    stats: Dict[str, Any] = defaultdict(list)
    successes = None

    pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)

    videos = []

    for i in pbar:
        observation, done = env.reset(), False
        trajectory_history = Trajectory.create_empty(observation_space, action_space)
        if len(videos) < num_videos:
            try:
                imgs = [env.sim.render(64, 48, camera_name="track")[::-1]]
            except:
                imgs = [env.render()[::-1]]

        eval_steps = 0
        while not done and eval_steps < max_steps:
            eval_steps += 1
            action = sample_actions(observation, trajectory_history)
            action = np.clip(action, -1, 1)
            new_observation, reward, done, info = env.step(action)
            if reward > 0.0:
                done = True
            # print("reward", reward)
            # env.render()
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
            for k in stats.keys():
                stats[k].append(float(info["episode"][k]))
                if verbose:
                    print(f"{k}: {info['episode'][k]}")

            ret = info["episode"]["return"]
            mean = np.mean(stats["return"])
            pbar.set_description(f"iter={i}\t last={ret:.2f} mean={mean}")
            if "is_success" in info:
                if successes is None:
                    successes = 0.0
                successes += info["is_success"]
        else:
            stats["return"].append(trajectory_history.rewards.sum())
            stats["length"].append(len(trajectory_history.rewards))
            # stats["achieved"].append(int(info["goal_achieved"]))

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


class OldGymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        # robots = "".join(
        #     [type(robot.robot_model).__name__ for robot in self.env.robots]
        # )
        robots = "panda"
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, 1)  # (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        # low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)  # TODO: all fake

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()


if __name__ == "__main__":
    dataset = MIMICDataset()

    dataset.env.reset()

    for i in range(1000):
        action = np.random.randn(7)  # sample random action
        obs, reward, _, info = dataset.env.step(
            action
        )  # take action in the environment
        print(obs)
        # dataset.env.render()  # render on display
