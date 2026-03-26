import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import FrameStackObservation


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


class FrozenJointsWrapper(gym.ActionWrapper):
    """Zero-out specified actuator indices and expose only the remaining ones.
    The agent sees a smaller Box action space (only the free joints)."""

    def __init__(self, env, frozen_indices):
        super().__init__(env)
        n_actuators = env.action_space.shape[0]
        self.frozen = np.array(sorted(frozen_indices), dtype=int)
        self.free = np.array(
            [i for i in range(n_actuators) if i not in self.frozen], dtype=int
        )
        low = env.action_space.low[self.free]
        high = env.action_space.high[self.free]
        self.action_space = Box(low=low, high=high, dtype=env.action_space.dtype)
        print(
            f"FrozenJointsWrapper: {len(self.frozen)} frozen, "
            f"{len(self.free)} free actuators"
        )

    def action(self, action):
        full = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        full[self.free] = action
        return full


class RenderGrayscaleWrapper(gym.ObservationWrapper):
    """Replace obs with grayscale render in a single wrapper.
    Expects env created with render_mode='rgb_array' and width/height
    already set to target resolution so no resize is needed."""

    def __init__(self, env, obs_size=84):
        super().__init__(env)
        self._obs_size = obs_size
        self.observation_space = Box(
            low=0, high=255, shape=(obs_size, obs_size), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if gray.shape != (self._obs_size, self._obs_size):
            gray = cv2.resize(
                gray, (self._obs_size, self._obs_size), interpolation=cv2.INTER_AREA
            )
        return gray


class EvalRenderWrapper(gym.Wrapper):
    """High-quality display for eval using a separate MuJoCo renderer at native
    resolution, independent of the 84x84 model observation pipeline.
    Falls back to upscaled env.render() for non-MuJoCo envs."""

    def __init__(self, env, display_size=480, window_name="Eval"):
        super().__init__(env)
        self._display_size = display_size
        self._window_name = window_name
        self._mj_renderer = None
        self._camera = None
        self._episode_frames = []
        self._is_mujoco = hasattr(env.unwrapped, "model") and hasattr(
            env.unwrapped, "data"
        )

    def _init_mj_renderer(self):
        if self._mj_renderer is not None:
            return
        import mujoco

        unwrapped = self.env.unwrapped
        # Ensure the model's offscreen framebuffer is large enough
        unwrapped.model.vis.global_.offwidth = max(
            unwrapped.model.vis.global_.offwidth, self._display_size
        )
        unwrapped.model.vis.global_.offheight = max(
            unwrapped.model.vis.global_.offheight, self._display_size
        )
        self._mj_renderer = mujoco.Renderer(
            unwrapped.model,
            height=self._display_size,
            width=self._display_size,
        )
        # Copy the exact camera from the env's own viewer (already configured
        # with default_cam_config + correct type). This guarantees the eval
        # display matches the same viewpoint the model was trained on.
        # Trigger viewer creation if it hasn't been created yet (state envs).
        mj_rend = getattr(unwrapped, "mujoco_renderer", None)
        if mj_rend is not None and getattr(mj_rend, "viewer", None) is None:
            mj_rend.render("rgb_array")
        viewer = getattr(mj_rend, "viewer", None) if mj_rend else None
        if viewer is not None and hasattr(viewer, "cam"):
            src = viewer.cam
            self._camera = mujoco.MjvCamera()
            self._camera.type = src.type
            self._camera.fixedcamid = src.fixedcamid
            self._camera.trackbodyid = src.trackbodyid
            self._camera.distance = src.distance
            self._camera.azimuth = src.azimuth
            self._camera.elevation = src.elevation
            self._camera.lookat[:] = src.lookat
        else:
            self._camera = -1  # free camera fallback

    def _get_display_frame(self):
        if self._is_mujoco:
            self._init_mj_renderer()
            self._mj_renderer.update_scene(
                self.env.unwrapped.data, camera=self._camera
            )
            return self._mj_renderer.render()
        # Fallback: upscale the low-res env render
        frame = self.env.render()
        if frame is not None:
            h, w = frame.shape[:2]
            if h != self._display_size or w != self._display_size:
                frame = cv2.resize(
                    frame,
                    (self._display_size, self._display_size),
                    interpolation=cv2.INTER_LINEAR,
                )
        return frame

    def _show(self, frame):
        if frame is None:
            return
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, img_bgr)
        cv2.waitKey(1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._get_display_frame()
        if frame is not None:
            self._episode_frames.append(frame)
        self._show(frame)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._episode_frames = []
        result = self.env.reset(**kwargs)
        frame = self._get_display_frame()
        if frame is not None:
            self._episode_frames.append(frame)
        self._show(frame)
        return result

    def get_episode_frames(self):
        return self._episode_frames

    def save_video(self, path, fps=None):
        """Save recorded episode frames as H.264 MP4 via imageio-ffmpeg."""
        if not self._episode_frames:
            return
        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        import imageio.v3 as iio

        iio.imwrite(path, self._episode_frames, fps=fps, codec="libx264")

    def close(self):
        if self._mj_renderer is not None:
            self._mj_renderer.close()
            self._mj_renderer = None
        cv2.destroyAllWindows()
        return self.env.close()


def make_state_env(env_id, render=False, seed=42, frozen_joints=None):

    render_mode = "rgb_array" if render else None

    if "Walker2d-v5" in env_id or "Hopper-v5" in env_id or "Humanoid-v5" in env_id:
        env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=20000,
            forward_reward_weight=0.5,
            ctrl_cost_weight=0.1,
            healthy_reward=1.25,
        )
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if frozen_joints:
        env = FrozenJointsWrapper(env, frozen_joints)

    # Discretize actions if needed
    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    if render:
        env = EvalRenderWrapper(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_pixel_env(env_id, render=False, seed=42, frozen_joints=None):
    """
    Creates env with Pixel observation + Discretization + Stack.
    Renders directly at 84x84 to avoid expensive high-res render + resize,
    and to keep the observation distribution identical between train and eval.
    The display wrapper (OpenCVRenderWrapper) upscales with smooth interpolation.
    """
    obs_size = 84

    if "Walker2d-v5" in env_id or "Hopper-v5" in env_id or "Humanoid-v5" in env_id:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            width=obs_size,
            height=obs_size,
            max_episode_steps=2000,
            forward_reward_weight=0.5,
            ctrl_cost_weight=0.1,
            healthy_reward=1.25,
        )
    else:
        env = gym.make(env_id, render_mode="rgb_array", width=obs_size, height=obs_size)

    if frozen_joints:
        env = FrozenJointsWrapper(env, frozen_joints)

    # Single wrapper: render → grayscale (no resize needed, already 84x84)
    env = RenderGrayscaleWrapper(env, obs_size=obs_size)

    # Discretize actions only for continuous-control envs (MuJoCo Box).
    # For discrete-action envs (e.g., CartPole), keep the original Discrete action space.
    if isinstance(env.action_space, Box):
        env = DiscretizedActionWrapper(env, bins=3)

    # Stack frames
    env = FrameStackObservation(env, stack_size=4)

    if render:
        env = EvalRenderWrapper(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_env(env_id, obs_type, render=False, seed=42, frozen_joints=None):
    if obs_type == "pixel":
        return make_pixel_env(env_id, render, seed, frozen_joints)
    elif obs_type == "state":
        return make_state_env(env_id, render, seed, frozen_joints)
    else:
        raise ValueError(f"Unsupported obs_type: {obs_type}")


