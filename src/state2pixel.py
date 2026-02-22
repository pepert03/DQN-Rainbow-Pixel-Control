"""state2pixel.py

Goal
----
Given a *trained* state-based DQN (teacher), learn a convolutional encoder that
maps rendered RGB frames -> state vectors. The DQN is kept frozen; only the CNN
is trained.

This is useful if you already have a good state-policy and you want a pixel
front-end without re-training the full DQN.

Usage (Windows)
--------------
uv run python .\src\state2pixel.py humanoid --train-steps 200000

The script expects:
- configs/hyperparameters.yml contains the preset (e.g. "humanoid")
- runs/<preset>.pt is a trained *state* DQN checkpoint
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

import gymnasium as gym

from dqn import DQN
from wrappers import DiscretizedActionWrapper, WalkerReward


def _load_preset(preset_name: str, config_path: str = "./configs/hyperparameters.yml") -> dict:
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with open(config_path, "r", encoding="utf-8") as f:
		all_config = yaml.safe_load(f) or {}
	if preset_name not in all_config:
		raise KeyError(
			f"Preset '{preset_name}' not found in {config_path}. Available: {list(all_config.keys())}"
		)
	cfg = all_config[preset_name]
	if not isinstance(cfg, dict):
		raise TypeError(f"Preset '{preset_name}' must be a mapping.")
	return cfg


def make_state_rgb_env(env_id: str, seed: int = 42, bins: int = 3) -> gym.Env:
	"""State observations + rgb_array rendering.

	We need state observations as labels, and RGB frames as inputs.
	Gymnasium requires render_mode at creation time.
	"""
	if "Walker2d-v5" in env_id:
		env = gym.make(env_id, render_mode="rgb_array")
		env = WalkerReward(env)
	else:
		env = gym.make(env_id, render_mode="rgb_array")

	# Match training: discretize continuous actions
	if isinstance(env.action_space, gym.spaces.Box):
		env = DiscretizedActionWrapper(env, bins=bins)

	env.action_space.seed(seed)
	env.observation_space.seed(seed)
	return env


class PixelToStateEncoder(nn.Module):
	"""CNN that predicts a state vector from a single RGB frame."""

	def __init__(self, state_dim: int):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, 8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
		)
		self.head = nn.Sequential(
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, state_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, 3, 84, 84] in [0, 255]
		x = x / 255.0
		z = self.conv(x)
		return self.head(z)


class PixelToStateDQN(nn.Module):
	"""pixels -> (encoder) -> predicted_state -> (frozen state DQN) -> Q-values."""

	def __init__(self, encoder: PixelToStateEncoder, dqn: DQN):
		super().__init__()
		self.encoder = encoder
		self.dqn = dqn

	def forward(self, pixels_bchw: torch.Tensor) -> torch.Tensor:
		pred_state = self.encoder(pixels_bchw)
		return self.dqn(pred_state)


@dataclass
class Replay:
	capacity: int
	state_dim: int
	image_h: int = 84
	image_w: int = 84

	def __post_init__(self):
		self._idx = 0
		self._size = 0
		self.images = np.zeros((self.capacity, self.image_h, self.image_w, 3), dtype=np.uint8)
		self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)

	@property
	def size(self) -> int:
		return self._size

	def add(self, image_hwc_uint8: np.ndarray, state: np.ndarray) -> None:
		self.images[self._idx] = image_hwc_uint8
		self.states[self._idx] = state
		self._idx = (self._idx + 1) % self.capacity
		self._size = min(self._size + 1, self.capacity)

	def sample(self, batch_size: int, device: torch.device):
		idxs = np.random.randint(0, self._size, size=batch_size)
		imgs = self.images[idxs]  # [B, H, W, 3]
		sts = self.states[idxs]

		# -> torch: [B, 3, H, W]
		imgs_t = torch.from_numpy(imgs).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
		sts_t = torch.from_numpy(sts).to(device=device, dtype=torch.float32)
		return imgs_t, sts_t


def _get_resized_frame(env: gym.Env, size=(84, 84)) -> np.ndarray:
	frame = env.render()  # HWC uint8
	if frame is None:
		raise RuntimeError("env.render() returned None; make sure render_mode='rgb_array'")
	if frame.shape[0] != size[0] or frame.shape[1] != size[1]:
		frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
	return frame


def evaluate_pixel_policy(
	env: gym.Env,
	pixel_policy: PixelToStateDQN,
	device: torch.device,
	episodes: int = 3,
	seed: int = 123,
) -> list[float]:
	pixel_policy.eval()
	returns: list[float] = []
	for ep in range(episodes):
		obs, _ = env.reset(seed=seed + ep)
		terminated = truncated = False
		ep_ret = 0.0

		# also add the initial observation/frame
		while not (terminated or truncated):
			frame = _get_resized_frame(env)
			x = torch.from_numpy(frame).to(device=device, dtype=torch.float32)
			x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,84,84]
			with torch.no_grad():
				action = pixel_policy(x).argmax(dim=1).item()
			obs, reward, terminated, truncated, _ = env.step(action)
			ep_ret += float(reward)
		returns.append(ep_ret)
	return returns


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Train a CNN encoder to predict states from pixels, using a frozen state-DQN as teacher."
		)
	)
	parser.add_argument("preset", help="Key in configs/hyperparameters.yml (must be a state-based DQN).")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--bins", type=int, default=3, help="Action discretization bins (must match state-DQN training).")
	parser.add_argument("--train-steps", type=int, default=200_000)
	parser.add_argument("--dataset-size", type=int, default=50_000)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--update-every", type=int, default=1)
	parser.add_argument("--warmup", type=int, default=2_000)
	parser.add_argument("--eval-every", type=int, default=10_000)
	parser.add_argument("--eval-episodes", type=int, default=3)
	parser.add_argument(
		"--teacher", 
		default=None,
		help="Path to trained state-DQN .pt. Defaults to runs/<preset>.pt",
	)
	parser.add_argument(
		"--out-encoder",
		default=None,
		help="Where to save encoder weights. Defaults to runs/<preset>_state2pixel_encoder.pt",
	)
	parser.add_argument(
		"--out-combined",
		default=None,
		help="Where to save combined model (encoder+dqn). Defaults to runs/<preset>_state2pixel_combined.pt",
	)
	args = parser.parse_args()

	cfg = _load_preset(args.preset)
	env_id = cfg.get("env_id")
	if not env_id:
		raise KeyError(f"Preset '{args.preset}' is missing env_id")

	obs_type = cfg.get("obs")
	if obs_type != "state":
		print(
			f"WARNING: preset '{args.preset}' has obs='{obs_type}'. "
			"This script expects a state-trained DQN checkpoint."
		)

	runs_dir = "runs"
	os.makedirs(runs_dir, exist_ok=True)

	teacher_path = args.teacher or os.path.join(runs_dir, f"{args.preset}.pt")
	out_encoder = args.out_encoder or os.path.join(runs_dir, f"{args.preset}_state2pixel_encoder.pt")
	out_combined = args.out_combined or os.path.join(runs_dir, f"{args.preset}_state2pixel_combined.pt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	print(f"Env: {env_id}")
	print(f"Teacher checkpoint: {teacher_path}")

	env = make_state_rgb_env(env_id, seed=args.seed, bins=args.bins)
	state_dim = int(np.prod(env.observation_space.shape))
	action_dim = int(env.action_space.n)

	# Load teacher (state) DQN
	teacher = DQN(state_dim, action_dim).to(device)
	teacher.load_state_dict(torch.load(teacher_path, map_location=device))
	teacher.eval()
	for p in teacher.parameters():
		p.requires_grad = False

	encoder = PixelToStateEncoder(state_dim).to(device)
	pixel_policy = PixelToStateDQN(encoder, teacher).to(device)

	optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
	loss_fn = nn.MSELoss()

	replay = Replay(capacity=args.dataset_size, state_dim=state_dim)

	obs, _ = env.reset(seed=args.seed)
	# store an initial (frame, state) pair
	replay.add(_get_resized_frame(env), np.asarray(obs, dtype=np.float32))

	losses = []
	terminated = truncated = False

	for step in range(1, args.train_steps + 1):
		if terminated or truncated:
			obs, _ = env.reset(seed=args.seed + step)
			terminated = truncated = False
			replay.add(_get_resized_frame(env), np.asarray(obs, dtype=np.float32))

		# Use teacher policy to generate actions (better coverage than random)
		obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
		with torch.no_grad():
			action = teacher(obs_t).argmax(dim=1).item()

		next_obs, reward, terminated, truncated, info = env.step(action)
		obs = next_obs

		frame = _get_resized_frame(env)
		replay.add(frame, np.asarray(obs, dtype=np.float32))

		if replay.size >= max(args.warmup, args.batch_size) and (step % args.update_every == 0):
			x, y = replay.sample(args.batch_size, device)
			pred = encoder(x)
			loss = loss_fn(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			losses.append(float(loss.item()))

		if args.eval_every > 0 and (step % args.eval_every == 0):
			avg_loss = float(np.mean(losses[-200:])) if losses else float("nan")
			rets = evaluate_pixel_policy(env, pixel_policy, device, episodes=args.eval_episodes, seed=1000)
			print(
				f"Step {step}/{args.train_steps} | loss={avg_loss:.6f} | eval_return(mean)={float(np.mean(rets)):.2f}"
			)
			torch.save(encoder.state_dict(), out_encoder)
			torch.save(pixel_policy.state_dict(), out_combined)

	# Final save
	torch.save(encoder.state_dict(), out_encoder)
	torch.save(pixel_policy.state_dict(), out_combined)
	print(f"Saved encoder to: {out_encoder}")
	print(f"Saved combined model to: {out_combined}")
	env.close()


if __name__ == "__main__":
	main()

