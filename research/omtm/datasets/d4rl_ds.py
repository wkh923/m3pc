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

from research.jaxrl.datasets.d4rl_dataset import D4RLDataset
from research.jaxrl.utils import make_env
from research.omtm.datasets.sequence_dataset import SequenceDataset


def get_datasets(
    seq_steps: bool,
    env_name: int,
    seed: int = 0,
    use_reward: bool = True,
    discount: int = 1.5,
    train_val_split: float = 0.95,
):
    env = make_env(env_name, seed)
    d4rl_dataset = D4RLDataset(env)
    train_d, val_d = d4rl_dataset.train_validation_split(train_val_split)

    # hack to add env to class for eval
    train_d.env = env
    val_d.env = env

    train_dataset = SequenceDataset(
        train_d,
        discount=discount,
        sequence_length=seq_steps,
        use_reward=use_reward,
        name=env_name,
    )
    val_dataset = SequenceDataset(
        val_d,
        discount=discount,
        sequence_length=seq_steps,
        use_reward=use_reward,
        name=env_name,
    )
    return train_dataset, val_dataset, train_d


def main():
    # env_names = [
    #     "hopper-medium-v2",
    #     "hopper-medium-replay-v2",
    #     "hopper-medium-expert-v2",
    #     "hopper-expert-v2",
    #     "walker2d-medium-v2",
    #     "walker2d-medium-replay-v2",
    #     "walker2d-medium-expert-v2",
    #     "walker2d-expert-v2",
    #     "halfcheetah-medium-v2",
    #     "halfcheetah-medium-replay-v2",
    #     "halfcheetah-medium-expert-v2",
    #     "halfcheetah-expert-v2",
    # ]
    # for d in [0.99, 1, 1.5]:
    #     for e in env_names:
    #         train_dataset, val_dataset = get_datasets(32, e, discount=d)
    #         train_dataset.trajectory_statistics()
    env_name = "antmaze-umaze-v2"
    train_dataset, val_dataset, _ = get_datasets(32, env_name, discount=1.0)
    list = []
    for observation, reward in zip(
        train_dataset.observations_raw, train_dataset.rewards_raw
    ):
        if reward == 1:
            list.append(observation[:2])
            print(observation[:2])
    print("mean:", sum(list) / len(list))


if __name__ == "__main__":
    main()
