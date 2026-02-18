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
    def __init__(self, env, bins=3):
        super().__init__(env)

        # 1. Obtener límites originales
        low = self.env.action_space.low
        high = self.env.action_space.high
        n_dims = self.env.action_space.shape[0]

        # 2. Crear valores posibles por dimensión (escalados correctamente)
        # Para cada dimensión, creamos 'bins' puntos entre su low y su high
        values_per_dim = [np.linspace(low[i], high[i], bins) for i in range(n_dims)]

        # 3. Pre-calcular la matriz de acciones (Mucho más rápido que itertools en el step)
        # self.actions_grid será de forma (bins^n_dims, n_dims)
        self.actions_grid = np.array(
            list(itertools.product(*values_per_dim)), dtype=np.float32
        )

        # 4. Definir el nuevo espacio discreto
        self.action_space = Discrete(len(self.actions_grid))

    def action(self, action_index):
        # Acceso directo por índice en la matriz pre-calculada
        return self.actions_grid[action_index]


class ExtractObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, key="pixels"):
        super().__init__(env)
        self.key = key
        # Update the observation space to match the shape of the pixel data
        self.observation_space = env.observation_space[key]

    def observation(self, obs):
        # Extract only the pixel data from the observation dictionary
        return obs[self.key]


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.debug = debug
        self.max_x_position = 0
        self.ctrl_cost_weight = 0.001
        self.forward_reward_weight = 100.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_position = info["x_position"]
        reward = 0

        if x_position > self.max_x_position:
            reward += (
                x_position - self.max_x_position
            ) * self.forward_reward_weight  # Reward for forward progress
            self.max_x_position = x_position

        # penalize control cost
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        reward -= ctrl_cost

        if self.debug:
            print(
                f"Step reward: {reward:.3f} (x_position: {x_position:.3f}, ctrl_cost: {ctrl_cost:.3f})"
            )
        if terminated or truncated:
            self.max_x_position = 0  # Reset max position at the end of an episode
        return obs, reward, terminated, truncated, info


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

    env = RewardWrapper(env, debug=render)

    # Gymnasium v1.x replacement for PixelObservationWrapper
    # render_only=True makes the observation be the rendered frame
    env = AddRenderObservation(env, render_only=True)

    # ResizeObservation receives an image (Box), not a dict
    env = ResizeObservation(env, (84, 84))

    # # Extract pixel observations (dict -> array)
    # env = ExtractObsWrapper(env, key="pixels")

    # Discretize actions (Fixed class name here)
    env = DiscretizedActionWrapper(env, bins=2)

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
