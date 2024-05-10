import argparse
import d4rl
import gym
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="halfcheetah-medium-v2")
    args = parser.parse_args()

    env = gym.make(args.env_name)

    dataset = env.get_dataset()
    if "infos/qpos" not in dataset:
        raise ValueError("Only MuJoCo-based environments can be visualized")
    qpos = dataset["infos/qpos"]
    qvel = dataset["infos/qvel"]
    rewards = dataset["rewards"]
    actions = dataset["actions"]
    observations = dataset["observations"]

    print(
        f"qpos: {qpos.shape}, qvel: {qvel.shape}, rewards: {rewards.shape}, actions: {actions.shape}, observations: {observations.shape}"
    )

    # the reward is with 1000 * 1000
    # sum rewards every 1000 steps
    rewards = rewards.reshape(-1, 1000).sum(axis=1)
    # print the top 10 index of the minimum rewards
    print(rewards.argsort()[:1])
    focus_idx = rewards.argsort()[:1]

    # for start in focus_idx:
    #     print(f"start: {start}")
    #     env.reset()
    #     t = 1000 * start
    #     env.set_state(qpos[t], qvel[t])
    #     while t < 1001 * start:
    #         env.set_state(qpos[t], qvel[t])
    #         env.render()
    #         t += 1

    # plot observations for index 446
    import matplotlib.pyplot as plt

    plt.plot(observations[446 * 1000 : 447 * 1000, 0:2])
    # plt.show()

    rot_states = observations[446 * 1000 : 447 * 1000, :]

    rot_states[0:70, 0] = np.linspace(0, -0.5, 70)
    rot_states[0:70, 1] = np.linspace(0.5, -3.14, 70)

    rot_states[30:100] = rot_states[0:70]

    rot_states[0:30, 0:2] = rot_states[0:1, 0:2]

    # rot_states[300:370] = rot_states[0:70]

    rot_states[100:] = rot_states[100 - 1]

    plt.plot(rot_states[:, 0:2])

    # save it to a file
    open("research/zoo/observation-rot.txt", "w").close()
    np.savetxt("research/zoo/observation-rot.txt", rot_states)
    plt.show()
