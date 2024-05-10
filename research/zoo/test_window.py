import gym
import time
import numpy as np
from matplotlib import pyplot as plt

# from IPython import display

# Create the Hopper environment
env = gym.make("Walker2d-v3", terminate_when_unhealthy=False)
observation = env.reset()

# Simulate the environment

plt.ion()
fig = plt.figure()

# empty the file
open("research/zoo/observation.txt", "w").close()


for i in range(1000):
    env.render(
        mode="human", width=800, height=200, camera_id=-1
    )  # Render the environment to visualize
    action = env.action_space.sample()  # Take a random action
    # print("action: ", action)
    # action = np.array([-0.5, 0.3, -0.5, 0.5, 0.2, -0.5])
    action = np.array([-0.5, 0.2, 0.0, 0.5, 0.2, -0.3])  # good action for splits
    observation, reward, done, info = env.step(action)
    # pause for 1 second
    print("step: ", i)
    # save the observation to a file, and readable by numpy or torch
    # in total 1000 rows, dont contrain the brace
    with open("research/zoo/observation.txt", "a") as f:
        for i in range(17):
            f.write(str(observation[i]) + " ")
        f.write("\n")

    # env.render()

    # if i % 10 == 0:
    #     plt.clf()

    #     fig = plt.imshow(frame)
    #     plt.show(block=True)
    time.sleep(0.01)
    if done:
        print("done")
        break

env.close()  # Close the environment when done

#

x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
