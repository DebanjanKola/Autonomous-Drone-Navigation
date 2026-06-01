# Repository-adaptation: drone navigation using Q-learning
# This single file contains multiple modules separated by markers.
# Save each section into its own file under the same project tree.

"""Package marker for the custom gym environment."""

from gym.envs.registration import register

register(
    id='DroneGrid-v0',
    entry_point='gym_drone.envs:DroneGridEnv',
)


