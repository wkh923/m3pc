import gym
import time
import numpy as np
from matplotlib import pyplot as plt

# from IPython import display

# Create the Hopper environment
# env = gym.make("Walker2d-v3", terminate_when_unhealthy=False)
# observation = env.reset()

# Simulate the environment

# plt.ion()
# fig = plt.figure()

# empty the file
open("research/zoo/observation-wig.txt", "w").close()


import numpy as np

# Constants
head_height = 1.25  # length of the bar in meters
total_steps = 1000
total_wiggles = 2
amplitude = 0.1  # amplitude of the wiggle in radians
neg_v_offset = 0.1  # offset for the negative velocity # -0.1
neg_angle_offset = -0.3  # offset for the negative angle # -0.3

# Calculating the angular frequency
total_time = total_steps  # Assuming each step is one unit of time
angular_frequency = 2 * np.pi * total_wiggles / total_time  # omega = 2*pi*f

# Time array
t = np.linspace(0, total_time, total_steps)

# Head angle over time using sinusoidal function for wiggling
theta = -amplitude * np.sin(angular_frequency * t)

# Calculate the position of the head
head_x = head_height * np.sin(theta)
head_y = head_height * np.cos(theta)

# Calculate velocities of the head
velocity_head_x = np.gradient(head_x, t) + neg_v_offset
velocity_head_y = np.gradient(head_y, t)

# Calculate angular velocities
angular_velocity_head = np.gradient(theta, t)
angular_velocity_foot = angular_velocity_head  # opposite of head angular velocity

# State array: [head_x, head_y, head_angle, foot_joint_angle, velocity_head_x, velocity_head_y, angular_velocity_head, angular_velocity_foot]
states = np.column_stack(
    (
        head_y,
        theta + neg_angle_offset,
        np.zeros(total_steps),
        np.zeros(total_steps),
        theta + neg_angle_offset,
        velocity_head_x,
        velocity_head_y,
        angular_velocity_head + neg_angle_offset,
        np.zeros(total_steps),
        np.zeros(total_steps),
        angular_velocity_foot + neg_angle_offset,
    )
)

# # Plotting all variables
# plt.plot(t, states)
# plt.legend(
#     [
#         "head_x",
#         "head_y",
#         "theta",
#         "foot_joint_angle",
#         "velocity_head_x",
#         "velocity_head_y",
#         "angular_velocity_head",
#         "angular_velocity_foot",
#     ]
# )
# # show the plot
# plt.show()

# Save the states to a file
np.savetxt("research/zoo/observation-wig.txt", states)

print(
    "State Shape:", states.shape
)  # Should be (1000, 8) where 8 is the number of state variables
