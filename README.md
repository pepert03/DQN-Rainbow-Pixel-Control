
# DeepRL Visual Locomotion: Control from Pixels in MuJoCo

> **A Deep Reinforcement Learning framework for training agents in continuous control physics simulations using high-dimensional visual observations.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics-red)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project Overview

This project explores the capabilities of **Deep Reinforcement Learning (DRL)** to solve complex motor control and locomotion tasks. Unlike standard control benchmarks that provide low-dimensional state vectors (joint angles, velocities), this framework challenges agents to learn policies **solely from raw RGB pixel observations**.

The system is built upon **MuJoCo** (Multi-Joint dynamics with Contact) and **DeepMind Control Suite**, utilizing the **Gymnasium** interface for standardization.

Key objectives include:
1.  **Visual Representation Learning:** Training CNN encoders to extract spatial features from physics simulations.
2.  **Algorithm Implementation:** deploying **Deep Q-Networks (DQN)** and **Rainbow DQN**.
3.  **Complex Behaviors:** Achieving stable locomotion and dynamic obstacle avoidance.

---

## Theoretical Background: Deep Q-Learning

The core algorithm driving this project is the **Deep Q-Network (DQN)**, a model-free, off-policy algorithm that approximates the optimal action-value function $Q^*(s, a)$.

### 1. The Bellman Optimality Equation
In Reinforcement Learning, the goal is to maximize the expected cumulative reward. The optimal Q-value is defined as the maximum expected return starting from state $s$, taking action $a$, and following the optimal policy thereafter:

$$Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]$$

Where:
* $s, s'$: Current and next states (high-dimensional pixel arrays).
* $a, a'$: Actions from the discrete action space $\mathcal{A}$.
* $r$: Immediate reward received.
* $\gamma$: Discount factor $\in [0, 1]$.

### 2. Function Approximation
Since the state space (RGB images) is too large for tabular Q-learning, we approximate $Q(s, a)$ using a deep neural network with weights $\theta$:
$$Q(s, a; \theta) \approx Q^*(s, a)$$

The policy $\pi$ is implicitly defined by selecting the action with the highest estimated value (greedy strategy):
$$a_t = \arg \max_{a} Q(s_t, a; \theta)$$

### 3. Network Architecture
The architecture follows the standard Mnih et al. proposal, adapted for continuous control tasks:
* **Input:** Stacked RGB frames ($84 \times 84 \times k$) to capture temporal dynamics.
* **Visual Encoder:** A Convolutional Neural Network (CNN) that processes raw pixels into a latent feature vector.
* **Q-Head:** A Multi-Layer Perceptron (MLP) that maps latent features to Q-values for each discrete action.

### 4. Loss Function and Optimization
The network is trained by minimizing the **Temporal Difference (TD) error**. The loss function $L(\theta)$ at iteration $i$ is the Expectation of the squared error between the prediction and the target:

$$L_i(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim U(D)} \left[ \left( y_i - Q(s, a; \theta_i) \right)^2 \right]$$

Where the **TD Target** $y_i$ is calculated using a separate **Target Network** ($\theta^-$) to stabilize training:

$$y_i = r + \gamma \max_{a'} Q(s', a'; \theta_{i}^-)$$

### 5. Stability Mechanisms
To handle the instability of non-linear function approximation, this implementation includes:
* **Experience Replay Buffer ($D$):** Stores transitions $(s, a, r, s')$ to break temporal correlations in the training data.
* **Target Network:** A frozen copy of the main network, updated periodically, to prevent the "chasing a moving target" problem.

---

## Rainbow DQN Integration

To improve sample efficiency and stability, this project also implements **Rainbow DQN**, which aggregates state-of-the-art extensions:

1.  **Double DQN:** Decouples action selection from evaluation to reduce Q-value overestimation.
    * $y^{Double} = r + \gamma Q(s', \arg \max_a Q(s', a; \theta); \theta^-)$
2.  **Prioritized Experience Replay (PER):** Samples transitions with high TD errors more frequently.
3.  **Dueling Architecture:** Splits the network into Value $V(s)$ and Advantage $A(s, a)$ streams.
    * $Q(s, a) = V(s) + (A(s, a) - \frac{1}{|\mathcal{A}|}\sum A(s, a'))$
4.  **Noisy Nets:** Replaces $\epsilon$-greedy exploration with learnable parametric noise in the weights.
5.  **Multi-step Learning ($n$-step):** Bootstraps from $n$ steps ahead to propagate rewards faster.
6.  **Distributional RL (C51):** Models the full distribution of returns rather than just the mean.

---

## Environments and Tasks

The project utilizes `dm_control` suites wrapped via `shimmy` and `gymnasium`.

### Supported Domains
* **Simple Locomotion (ES):** `Walker-Walk`, `Hopper-Hop`, `Cheetah-Run`.
    * *Challenge:* Balance and forward momentum.
* **Complex Control (EC):** `Humanoid-Walk`.
    * *Challenge:* High-dimensional control and stability.

### Advanced Tasks
* **Obstacle Avoidance:** A custom variation where the agent must navigate corridors and avoid static or dynamic obstacles while maintaining locomotion.

---

## Installation & Usage

### Prerequisites
* Python 3.10+
* `uv` (recommended for dependency management) or `pip`

### Setup
```bash
# Clone the repository
git clone [https://github.com/username/deeprl-mujoco-locomotion.git](https://github.com/username/deeprl-mujoco-locomotion.git)
cd deeprl-mujoco-locomotion

# Install dependencies
pip install -r requirements.txt
# OR using uv
uv sync

```

### Training

To train a baseline DQN agent on the Walker task:

```bash
python train.py \
    --algo dqn \
    --env dm_control/walker-walk \
    --visual-obs \
    --total-timesteps 1000000

```

To train a Rainbow agent with specific ablation (e.g., Double DQN + PER):

```bash
python train.py \
    --algo rainbow \
    --env dm_control/cheetah-run \
    --use-double \
    --use-per

```

### Evaluation

To visualize the agent's performance and save a video:

```bash
python evaluate.py \
    --model-path checkpoints/best_model.pt \
    --render-mode rgb_array \
    --save-video

```

---

## Methodology & Experiments

The project is divided into distinct experimental phases:

1. 
**DQN Baseline:** Establishing a performance baseline for visual locomotion.


2. 
**Rainbow Benchmarking:** Comparing convergence speed and asymptotic performance against DQN.


3. 
**Ablation Studies:** Analyzing the individual contribution of Rainbow components (e.g., Dueling vs. Noisy Nets).


4. 
**Robustness Analysis:** Testing the agent in the Obstacle Avoidance scenario.



## Citation

If you use this code for your research, please link back to this repository.
Original methodology based on the "Deep Reinforcement Learning Lab" guidelines.