import gym
import time
import numpy as np
from matplotlib import pyplot as plt

# get file path dictionary
import os
import sys


# Constants
head_height = 1.25  # length of the bar in meters
total_steps = 1000

amplitude = 0.05  # amplitude of the wiggle in radians
neg_v_offset = 0.0  # offset for the negative velocity # -0.1
neg_angle_offset = -0.0  # offset for the negative angle # -0.3

for total_wiggles in [0.001, 2, 6]:  # different wiggle frequencies for hopper
    if total_wiggles == 0.001:
        neg_angle_offset = -0.02
    else:
        neg_angle_offset = -0.0

    open(
        os.path.join(os.path.dirname(__file__), f"hopper-wiggle-f{total_wiggles}.txt"),
        "w",
    ).close()

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
    velocity_head_x = np.gradient(head_x, t) * 10 + neg_v_offset
    velocity_head_y = np.gradient(head_y, t) * 10

    # Calculate angular velocities
    angular_velocity_head = np.gradient(theta, t) * 10
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
    np.savetxt(
        os.path.join(os.path.dirname(__file__), f"hopper-wiggle-f{total_wiggles}.txt"),
        states,
    )

# load generated waypoints and visualize
waypoints_1 = np.loadtxt(
    os.path.join(os.path.dirname(__file__), "hopper-wiggle-f2.txt")
)
waypoints_2 = np.loadtxt(os.path.join(os.path.dirname(__file__), "cheeta-flip.txt"))
waypoints_3 = np.loadtxt(os.path.join(os.path.dirname(__file__), "walker-splits.txt"))

# show it in a 1x3 subplot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(waypoints_1[:, 0:2])
axs[0].set_title("hopper-wiggle-f2")
axs[0].legend(["Head-Y", "Body-Angle"])

axs[1].plot(waypoints_2[:200, 0:2])
axs[1].set_title("cheeta-flip")
axs[1].legend(["Head-Y", "Body-Angle"])

axs[2].plot(waypoints_3[:200, [0, 2]])
axs[2].set_title("walker-splits")
axs[2].legend(["Head-Y", "Left-Leg-Angle"])

plt.show()
