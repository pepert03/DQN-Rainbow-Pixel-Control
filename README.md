# Deep Reinforcement Learning: Locomotion from Pixels

Implementation of **DQN** and **Rainbow DQN** for continuous control tasks in **MuJoCo** environments, using **pixel-based observations** instead of state vectors.

## Objectives
1.  Train a DQN agent to solve locomotion tasks (e.g., Walker, Cheetah) from raw RGB images.
2.  Implement **Rainbow DQN** improvements (Double Q-Learning, PER, Dueling Networks, etc.).
3.  Compare the performance and sample efficiency of both algorithms.
4.  Solve complex tasks involving obstacle avoidance.

## Tech Stack
* **Engine:** [MuJoCo](https://github.com/google-deepmind/mujoco) via [DeepMind Control Suite](https://github.com/google-deepmind/dm_control).
* **Interface:** [Gymnasium](https://gymnasium.farama.org/) (via Shimmy wrapper).
* **Framework:** PyTorch.
* **Manager:** `uv`.

## Project Overview

This project addresses one of the fundamental challenges in Deep Reinforcement Learning (Deep RL): continuous motor control from high-dimensional observations. unlike traditional approaches that rely on low-dimensional state vectors (position, velocity, joint angles), this system learns control policies based **exclusively on RGB images** of the environment.

The core of the project focuses on the implementation, benchmarking, and analysis of **Deep Q-Network (DQN)** versus its advanced variant, **Rainbow DQN**. The objective is to train simulated agents (such as *Walkers* or *Humanoids*) to achieve stable locomotion and navigate obstacles within a realistic physics engine.

## Theoretical Foundation: Deep Q-Network (DQN)

The central algorithm employed is **DQN**, a *model-free*, *off-policy* method that combines classical Q-Learning with non-linear function approximators (Deep Neural Networks).

### 1. The Objective: Action-Value Function

The agent's goal is to maximize the expected cumulative future reward. To achieve this, we approximate the optimal action-value function , which represents the maximum expected return starting from state , taking action , and following the optimal policy thereafter:

Where:

* 
: State space (RGB pixel frames).


* 
: Discretized action space.


* : Immediate reward at time .
* : Discount factor.

### 2. The Bellman Optimality Equation

The mathematical backbone of DQN is the **Bellman Optimality Equation**, which establishes a recursive relationship for :

### 3. Neural Approximation

Since the state space (pixels) is too large for a tabular Q-function, we utilize a Deep Neural Network with weights  to approximate the function: .

#### Model Architecture

The system implements a **CNN + MLP** architecture designed to extract visual features and determine control actions:

1. **Visual Encoder:** A Convolutional Neural Network (CNN) processes the RGB input (potentially with frame stacking) to extract a latent feature vector.
2. **Control Head:** Fully Connected (Dense) layers take the latent vector and estimate Q-values for each discrete action.
3. **Output:** A vector of Q-values, where the policy is derived implicitly via a greedy approach:



.



### 4. Loss Function and Optimization

The network is trained iteratively by minimizing the Mean Squared Error (MSE) between the current Q-network prediction and a "target" value derived from the Bellman equation.

The loss function  at iteration  is defined as:

Where the Temporal Difference (TD) target  is calculated using a separate **Target Network** () to stabilize training:

### 5. Stability Mechanisms

To achieve convergence in non-linear environments, two key techniques are employed:

* **Experience Replay Buffer ():** Stores transitions  and samples random minibatches to break temporal correlations between consecutive data points.
* **Target Network ():** A copy of the Q-network that is updated slowly (via soft updates or periodic hard updates), preventing the target  from shifting constantly while the main network is updated.

---

## Project Features

This repository includes modular implementations to address varying levels of control complexity:

### 1. Simulation Environments

* 
**Physics Engine:** [MuJoCo](https://github.com/google-deepmind/mujoco) (Multi-Joint dynamics with Contact) for realistic dynamics.


* 
**Task Suite:** DeepMind Control Suite (`dm_control`), utilizing environments such as *Walker*, *Cheetah*, and *Humanoid*.


* 
**Interface:** Gymnasium, used to standardize the agent-environment interaction and handle pixel preprocessing.



### 2. Algorithms

* 
**DQN (Baseline):** Canonical implementation with action space discretization and visual encoding.


* 
**Rainbow DQN:** An integration of state-of-the-art extensions:


* *Double DQN* (Decoupling action selection and evaluation).
* *Prioritized Experience Replay (PER)* (Focusing on high TD-error transitions).
* *Dueling Network Architecture* ( decomposition).
* *Noisy Nets* for parametric exploration.
* *Distributional RL* (C51/Categorical DQN).
* *N-step Returns*.



### 3. Tasks

* 
**Visual Locomotion:** Learning stable gait and forward motion from scratch using only visual input.


* 
**Obstacle Avoidance:** Advanced agents capable of detecting and evading barriers while maintaining locomotion.



## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Simulation:** MuJoCo, dm_control
* **Interfaces:** Gymnasium, Shimmy
* **Dependency Management:** `uv`

## Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/deeprl-mujoco-locomotion.git
cd deeprl-mujoco-locomotion

```


2. **Install dependencies using `uv`:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

```


3. **Train an agent (e.g., DQN on Walker):**
```bash
python train.py --algo dqn --env walker-walk --visual-obs

```



## Performance Analysis

The project generates detailed telemetry, including cumulative reward curves and loss convergence graphs. The experimental results aim to demonstrate the superior sample efficiency and stability of Rainbow DQN compared to the baseline, particularly in high-dimensional control tasks like *Humanoid*.

