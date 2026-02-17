import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
    FrameStackObservation,
    HumanRendering,
)
import itertools


class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    Maps discrete action index to continuous force vector.
    Required because DQN outputs a single integer, but MuJoCo expects a vector of floats.
    """

    def __init__(self, env, bins=3):
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space

        # Get number of actuators (e.g., Walker2d: 6, Hopper: 3)
        n_actions = self.orig_action_space.shape[0]

        # Generate all possible force combinations [-1, 0, 1]
        self.actions_grid = list(itertools.product([-1, 0, 1], repeat=n_actions))

        # Define new discrete action space
        self.action_space = Discrete(len(self.actions_grid))

    def action(self, action_index):
        discrete_action = self.actions_grid[action_index]
        low = self.orig_action_space.low
        high = self.orig_action_space.high
        return np.array(discrete_action, dtype=np.float32) * high


class ExtractObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, key="pixels"):
        super().__init__(env)
        self.key = key
        # Update the observation space to match the shape of the pixel data
        self.observation_space = env.observation_space[key]

    def observation(self, obs):
        # Extract only the pixel data from the observation dictionary
        return obs[self.key]


def make_env(env_id, render=False, seed=42, **env_kwargs):
    """
    Creates env with Pixel observation + Discretization + Stack.
    This function returns a thunk for use with SyncVectorEnv.
    """
    # MuJoCo environments require render_mode to be set at creation time.
    # IMPORTANT: AddRenderObservation needs env.render() to return an RGB array,
    # which is not the case for render_mode="human".
    # So we always create the env with render_mode="rgb_array" for pixel obs.
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

    # Gymnasium v1.x replacement for PixelObservationWrapper
    # render_only=True makes the observation be the rendered frame
    env = AddRenderObservation(env, render_only=True)

    # ResizeObservation receives an image (Box), not a dict
    env = ResizeObservation(env, (84, 84))

    # # Extract pixel observations (dict -> array)
    # env = ExtractObsWrapper(env, key="pixels")

    # Discretize actions (Fixed class name here)
    env = DiscretizedActionWrapper(env, bins=3)

    # Stack frames
    env = FrameStackObservation(env, stack_size=4)

    # If the user wants to *see* the episode, wrap with HumanRendering at the end.
    # This avoids breaking AddRenderObservation's assertion.
    if render:
        env = HumanRendering(env)

    # # CleanRL wrappers
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # if capture_video and idx == 0:
    #     env = gym.wrappers.RecordVideo(
    #         env, f"videos/{run_name}", episode_trigger=lambda x: x % 1000 == 0
    #     )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
