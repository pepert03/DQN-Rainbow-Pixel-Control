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


# Discreize continuous actions n into bins, no need to separete into combinations of actions for each dimension, as the agent will just choose one action at a time
# Example: For HalfCheetah-v5, action space is Box(-1.0, 1.0, (6,), float32)
# With bins=3, we create discrete actions for each dimension: [-1.0, 0.0, 1.0]
# The total number of discrete actions becomes 3x6 = 18
# but the action 0 is the same for all dimensions so we need to remove duplicates
# The resulting action space is Discrete(13) with actions
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=3):
        super().__init__(env)

        # Nos aseguramos de que el entorno original sea continuo
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "Action space must be continuous (Box)."

        low = self.env.action_space.low
        high = self.env.action_space.high
        n_dims = self.env.action_space.shape[0]

        # 1. Empezamos la lista con la acción base: "No hacer nada" (Vector de ceros)
        actions = [np.zeros(n_dims, dtype=np.float32)]

        # 2. Iteramos por cada dimensión para crear sus acciones individuales
        for i in range(n_dims):
            # Generamos los valores posibles para esta articulación/motor
            values = np.linspace(low[i], high[i], bins)

            for v in values:
                # Evitamos añadir el 0.0 de nuevo, ya que está cubierto por la acción base
                if not np.isclose(v, 0.0):
                    # Creamos un vector de ceros y solo modificamos la dimensión actual
                    action_vec = np.zeros(n_dims, dtype=np.float32)
                    action_vec[i] = v
                    actions.append(action_vec)

        # Convertimos a array de numpy para acceso rápido en el step
        self.actions_grid = np.array(actions, dtype=np.float32)

        # 3. Definimos el nuevo espacio de acción discreto
        # Para bins=3 y n_dims=6, esto será Discrete(13)
        self.action_space = Discrete(len(self.actions_grid))
        print(
            f"DiscretizedActionWrapper initialized with {len(self.actions_grid)} discrete actions."
        )

    def action(self, action_index):
        # Mapea el entero que devuelve la DQN al vector continuo para MuJoCo
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

    if "Walker2d-v5" in env_id:
        env = gym.make(
            env_id,
            render_mode="human" if render else None,
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=1500,
        )
        env = WalkerReward(env)
    else:
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

    if "Walker2d-v5" in env_id:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            healthy_angle_range=(-0.4, 0.4),
            max_episode_steps=1500,
        )
        env = WalkerReward(env)
    else:
        env = gym.make(env_id, render_mode="rgb_array")

    # render_only=True makes the observation be the rendered frame
    env = AddRenderObservation(env, render_only=True)

    # ResizeObservation receives an image (Box), not a dict
    env = ResizeObservation(env, (84, 84))

    # Discretize actions (Fixed class name here)
    env = DiscretizedActionWrapper(env, bins=3)

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
