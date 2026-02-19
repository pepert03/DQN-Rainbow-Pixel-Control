
# DQN / Pixel-DQN in MuJoCo (Gymnasium)

This repository trains **DQN-style agents** on MuJoCo tasks using Gymnasium, with two observation modes:

- **state**: classic low-dimensional state vectors.
- **pixel**: visual observations from rendered RGB frames (resized + frame-stacked).

Training/evaluation is driven by presets in `configs/hyperparameters.yml`.

## Theoretical background (DQN)

The core idea of **Deep Q-Learning** is to approximate the optimal action-value function $Q^*(s, a)$ with a neural network $Q(s, a; \theta)$.

### Bellman optimality equation

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

Where:

- $s, s'$ are current/next observations (state vectors or pixel stacks).
- $a, a'$ are actions from a **discrete** action space.
- $r$ is the reward.
- $\gamma \in [0, 1]$ is the discount factor.

### TD target + loss

With a target network ($\theta^-$), the TD target is:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

and the DQN is trained by minimizing the mean-squared TD error:

$$L(\theta) = \mathbb{E}\left[(y - Q(s, a; \theta))^2\right]$$

### Stability mechanisms

This implementation follows the standard stabilizers:

- **Experience replay**: store transitions $(s, a, r, s', done)$ and sample mini-batches.
- **Target network**: periodically copy weights to make the target less non-stationary.

### Pixels: why the first Conv2D uses 12 channels

In pixel mode the env produces observations shaped like `[stack, H, W, C]`.
With `stack_size=4` and RGB (`C=3`), the effective input channels are $4 \times 3 = 12$.
That is why the CNN starts with `Conv2d(12, ...)`.

## Rainbow DQN note

TODO

## Repository structure

```
.
├─ configs/
│  └─ hyperparameters.yml      # presets (env_id, obs, hyperparameters)
├─ runs/                       # outputs: .pt models, logs, plots
├─ src/
│  ├─ agent.py                 # main entrypoint: train/eval
│  ├─ dqn.py                   # networks: DQN (state) + Pixel_DQN (CNN)
│  ├─ buffer.py                # simple replay buffer
│  ├─ wrappers.py              # wrappers + env factories (state/pixel)
│  ├─ buffers.py, dqn_mujoco.py # additional experiments (may be WIP)
│  └─ __init__.py
├─ mujoco_test.py              # minimal MuJoCo viewer sanity check
├─ wrapper_try.py              # minimal Gymnasium human-render example
├─ notebooks/                  # notebooks (exploration/debug)
└─ Report/                     # report/latex
```

### What gets saved to `runs/`?

`src/agent.py` saves:

- `runs/<preset>.pt`: best model so far (by episodic return).
- `runs/<preset>_checkpoint.pt`: periodic checkpoint.
- `runs/<preset>.log`: training log.
- `runs/<preset>.png`: plot (mean return + epsilon).

## Installation

### Requirements

- Python >= 3.10
- MuJoCo via `gymnasium[mujoco]` (installed as a dependency)

### `uv`

```bash
git clone https://github.com/pepert03/DQN-Rainbow-Pixel-Control
cd DQN-Rainbow-Pixel-Control

uv sync
```

## Usage

### 1) Choose a preset (env + obs + hyperparameters)

Edit `configs/hyperparameters.yml`. Examples:

- `walker2d`: `env_id: Walker2d-v5`, `obs: pixel`
- `humanoid`: `env_id: Humanoid-v5`, `obs: state`
- `cartpole`: `env_id: CartPole-v1`, `obs: state`

### 2) Training

From the repo root:

```bash
uv run python ./src/agent.py walker2d --train
```

### 3) Evaluation (with rendering)

Run the agent in evaluation mode (loads `runs/<preset>.pt`):

```bash
uv run python ./src/agent.py walker2d
```

Notes:

- In **pixel** mode, the wrapper creates the env with `render_mode="rgb_array"` to obtain frames, and optionally uses `HumanRendering` to display.
- In **state** mode, the env uses `render_mode="human"` when `render=True`.