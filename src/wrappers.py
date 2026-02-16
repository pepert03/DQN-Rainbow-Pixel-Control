import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import PixelObservationWrapper, ResizeObservation, FrameStack
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
    def __init__(self, env, key='pixels'):
        super().__init__(env)
        self.key = key
        # Update the observation space to match the shape of the pixel data
        self.observation_space = env.observation_space[key]

    def observation(self, obs):
        # Extract only the pixel data from the observation dictionary
        return obs[self.key]

def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Creates env with Pixel observation + Discretization + Stack.
    """
    def thunk():
        # Initialize environment via Shimmy (handles dm_control)
        env = gym.make(env_id, render_mode="rgb_array")

        # Get dictionary with pixels
        env = PixelObservationWrapper(env, pixels_only=True)

        # Extract the image from the dictionary
        env = ExtractObsWrapper(env, key='pixels')

        # ResizeObservation receives an image (Box), not a dict
        env = ResizeObservation(env, (84, 84))

        # Discretize actions (Fixed class name here)
        env = DiscretizedActionWrapper(env, bins=3)

        # Stack frames
        env = FrameStack(env, 4)

        # CleanRL wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 1000 == 0)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk