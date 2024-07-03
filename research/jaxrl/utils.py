# MIT License

# Copyright (c) 2023 Meta Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from research.jaxrl import wrappers


def make_env(
    env_name: str,
    seed: int,
    save_folder: Optional[str] = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    frame_stack: int = 1,
    from_pixels: bool = False,
    pixels_only: bool = True,
    image_size: int = 84,
    sticky: bool = False,
    gray_scale: bool = False,
    flatten: bool = True,
) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split("-")
        env = wrappers.DMCEnv(
            domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed}
        )

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == "quadruped" else 0
        env = PixelObservationWrapper(
            env,
            pixels_only=pixels_only,
            render_kwargs={
                "pixels": {
                    "height": image_size,
                    "width": image_size,
                    "camera_id": camera_id,
                }
            },
        )
        env = wrappers.TakeKey(env, take_key="pixels")
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def make_unseen_env(
    env_name: str,
    seed: int,
    save_folder: Optional[str] = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    frame_stack: int = 1,
    from_pixels: bool = False,
    pixels_only: bool = True,
    image_size: int = 84,
    sticky: bool = False,
    gray_scale: bool = False,
    flatten: bool = True,
) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    white = False
    name_corri_map = {
        "halfcheetah-medium-replay-v2": "HalfCheetah-v3",
        "hopper-medium-v2": "Hopper-v3",
        "walker2d-medium-v2": "Walker2d-v3",
    }

    if env_name in env_ids:
        # replace the "v2" in env_name with "v3" to get the unseen version, TODO: currently fix to Hopper-v3
        assert env_name in [
            "halfcheetah-medium-replay-v2",
            "hopper-medium-v2",
            "walker2d-medium-v2",
        ], "env_name not in the list"

        if white is False:
            if env_name == "halfcheetah-medium-replay-v2":
                env_name = "HalfCheetah-v3"
                env = gym.make(env_name)
            elif env_name == "hopper-medium-v2":
                env_name = "Hopper-v3"
                env = gym.make(env_name, terminate_when_unhealthy=False)
            else:
                env_name = "Walker2d-v3"
                env = gym.make(env_name, terminate_when_unhealthy=False)

        else:
            if env_name == "halfcheetah-medium-replay-v2":
                env_name = "HalfCheetah-v3"
                env = gym.make(
                    env_name,
                    xml_file="/home/hu/mtm/research/zeroshot_omtm/assets/white_cheeta.xml",
                )
            elif env_name == "hopper-medium-v2":
                env_name = "Hopper-v3"
                env = gym.make(
                    env_name,
                    terminate_when_unhealthy=False,
                    xml_file="/home/hu/mtm/research/zeroshot_omtm/assets/white_hopper.xml",
                )
            else:
                env_name = "Walker2d-v3"
                env = gym.make(
                    env_name,
                    terminate_when_unhealthy=False,
                    xml_file="/home/hu/mtm/research/zeroshot_omtm/assets/white_walker.xml",
                )
    else:
        domain_name, task_name = env_name.split("-")
        env = wrappers.DMCEnv(
            domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed}
        )

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == "quadruped" else 0
        env = PixelObservationWrapper(
            env,
            pixels_only=pixels_only,
            render_kwargs={
                "pixels": {
                    "height": image_size,
                    "width": image_size,
                    "camera_id": camera_id,
                }
            },
        )
        env = wrappers.TakeKey(env, take_key="pixels")
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
