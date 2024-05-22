import os
import json
import h5py
import numpy as np

import robomimic
import robomimic.utils.file_utils as FileUtils

# import robosuite as suite

# the dataset registry can be found at robomimic/__init__.py
from robomimic import DATASET_REGISTRY


dataset_path = "/home/hu/mtm/outputs/mimic/paired/low_dim_v141.hdf5"
# assert os.path.exists(dataset_path)

print("path", dataset_path)


import numpy as np

import torch
from torch.utils.data import DataLoader

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


dataset = SequenceDataset(
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
print(dataset)

print("path", dataset_path)

print(dataset.dataset_keys)

focus_keys = ["actions", "rewards", "dones", "obs"]  # ep is each length

list_dict = {key: [] for key in focus_keys}
obs_keys = dataset.obs_keys

for d in range(len(dataset.demos)):
    traj_dict = dataset.get_trajectory_at_index(d)
    # traj_dict is a dict with keys and array as values
    # concate all the obs_keys in traj_dict["obs"] as a numpy array

    np_obs = np.concatenate([traj_dict["obs"][key] for key in obs_keys], axis=1)
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
    # concatenate the nps in corrisponding dict keys
    list_dict["obs"].append(np_obs[: done_index + 1])
    list_dict["actions"].append(np_actions[: done_index + 1])
    list_dict["rewards"].append(np_rewards[: done_index + 1])
    list_dict["dones"].append(np_dones[: done_index + 1])


np_dict = {keys: np.concatenate(list_dict[keys], axis=0) for keys in focus_keys}

print(
    "all the shapes",
    np_dict["obs"].shape,
    np_dict["actions"].shape,
    np_dict["rewards"].shape,
    np_dict["dones"].shape,
)

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


# create environment instance
# env = suite.make(
#     env_name="PickPlaceCan",  # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     render_camera="frontview",  # ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand').
# )

print("path", dataset_path)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

# env_meta["env_kwargs"].update({"render_camera": None})

# print(env_meta)
print("path", dataset_path)
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    env_name=env_meta["env_name"],
    render=True,
    render_offscreen=False,
    use_image_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np_dict["actions"][i]  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    print(obs)
    env.render()  # render on display
    if np_dict["dones"][i]:
        env.reset()
