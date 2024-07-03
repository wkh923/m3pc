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

import collections

import gym
import numpy as np
from gym.spaces import Box


# From https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L229
# and modified for memory efficiency.
class LazyFrames(object):
    def __init__(self, frames, stack_axis=-1):
        self._frames = frames
        self._stack_axis = stack_axis

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=self._stack_axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stack_axis=-1, lazy=False):
        super().__init__(env)
        self._num_stack = num_stack
        self._stack_axis = stack_axis
        self._lazy = lazy

        self._frames = collections.deque([], maxlen=num_stack)

        low = np.repeat(self.observation_space.low, num_stack, axis=stack_axis)
        high = np.repeat(self.observation_space.high, num_stack, axis=stack_axis)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._num_stack
        if self._lazy:
            return LazyFrames(list(self._frames), stack_axis=self._stack_axis)
        else:
            return np.concatenate(list(self._frames), axis=self._stack_axis)
