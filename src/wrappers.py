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


class WalkerReward(gym.Wrapper):
    def __init__(self, env, torso_weight=1.0, knee_weight=0.3, symmetry_weight=0.1):
        super().__init__(env)
        self.torso_weight = torso_weight
        self.knee_weight = knee_weight
        self.symmetry_weight = symmetry_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        torso_angle = obs[1]
        left_knee = obs[3]
        right_knee = obs[6]
        left_hip = obs[2]
        right_hip = obs[5]

        torso_penalty = self.torso_weight * torso_angle**2
        knee_penalty = self.knee_weight * (left_knee**2 + right_knee**2)
        symmetry_penalty = self.symmetry_weight * (
            (left_knee - right_knee) ** 2 + (left_hip - right_hip) ** 2
        )

        # Shaped reward
        shaped_reward = reward - torso_penalty - knee_penalty - symmetry_penalty

        return obs, shaped_reward, terminated, truncated, info


def make_state_env(env_id, render=False, seed=42):
    env = gym.make(env_id, render_mode="human" if render else None)

    # Discretize actions if needed
    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_pixel_env(env_id, render=False, seed=42):
    """
    Creates env with Pixel observation + Discretization + Stack.
    This function returns a thunk for use with SyncVectorEnv.
    """
    # MuJoCo environments require render_mode to be set at creation time.
    # IMPORTANT: AddRenderObservation needs env.render() to return an RGB array,
    # which is not the case for render_mode="human".
    # So we always create the env with render_mode="rgb_array" for pixel obs.
    env = gym.make(env_id, render_mode="rgb_array")

    if "Walker2d-v5" in env_id:
        env = WalkerReward(env)
        print("Applied WalkerReward wrapper to Walker2d-v5")

    # render_only=True makes the observation be the rendered frame
    env = AddRenderObservation(env, render_only=True)

    # ResizeObservation receives an image (Box), not a dict
    env = ResizeObservation(env, (84, 84))

    # Discretize actions (Fixed class name here)
    env = DiscretizedActionWrapper(env, bins=2)

    # Stack frames
    env = FrameStackObservation(env, stack_size=4)

    if render:
        env = HumanRendering(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_env(env_id, obs_type, render=False, seed=42):
    if obs_type == "pixel":
        return make_pixel_env(env_id, render, seed)
    elif obs_type == "state":
        return make_state_env(env_id, render, seed)
    else:
        raise ValueError(f"Unsupported obs_type: {obs_type}")
