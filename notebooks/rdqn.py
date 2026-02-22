"""
Rainbow DQN — PyTorch + Gymnasium
==================================
Combines: Double DQN, PER, Dueling, Multi-step, C51, NoisyNet
"""

import math
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────────────────────────
# 1.  Segment Tree for Prioritized Replay
# ─────────────────────────────────────────────


class SegmentTree:
    """Minimal segment tree for sum & min queries."""

    def __init__(self, capacity: int, operation, init_value: float):
        self.capacity = capacity
        self.tree = [init_value] * (2 * capacity)
        self.operation = operation
        self.init_value = init_value

    def _operate(self, start: int, end: int, node: int, node_start: int, node_end: int):
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate(start, end, 2 * node, node_start, mid)
        elif start >= mid:
            return self._operate(start, end, 2 * node + 1, mid, node_end)
        else:
            return self.operation(
                self._operate(start, mid, 2 * node, node_start, mid),
                self._operate(mid, end, 2 * node + 1, mid, node_end),
            )

    def query(self, start: int, end: int):
        return self._operate(start, end, 1, 0, self.capacity)

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


class SumTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=lambda a, b: a + b, init_value=0.0)

    def sum(self, start: int = 0, end: int = None):
        if end is None:
            end = self.capacity
        return self.query(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find the highest index `i` in the tree such that
        sum(tree[0]..tree[i]) <= prefixsum."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = None):
        if end is None:
            end = self.capacity
        return self.query(start, end)


# ─────────────────────────────────────────────
# 2.  Prioritized Replay Buffer (with N-step)
# ─────────────────────────────────────────────


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with N-step return support."""

    def __init__(
        self,
        obs_dim: int,
        capacity: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        assert capacity & (capacity - 1) == 0, "capacity must be power of 2"

        self.obs_dim = obs_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha  # prioritisation exponent
        self.n_step = n_step
        self.gamma = gamma

        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

        # Storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Segment trees
        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)

        # N-step buffer
        self.n_step_buffer: Deque = deque(maxlen=n_step)

    def _get_n_step_info(self) -> Tuple[np.float32, np.ndarray, bool]:
        """Compute n-step return, next observation and done."""
        reward, next_obs, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        return reward, next_obs, done

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple:
        transition = (obs, action, reward, next_obs, done)
        self.n_step_buffer.append(transition)

        # Only store once we have enough for n-step
        if len(self.n_step_buffer) < self.n_step:
            return ()

        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.ptr
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        priority_alpha = self.max_priority**self.alpha
        self.sum_tree[idx] = priority_alpha
        self.min_tree[idx] = priority_alpha

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        return transition

    def sample(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Proportional prioritisation sampling."""
        assert self.size >= self.batch_size
        indices = self._sample_proportional()

        obs = self.obs[indices]
        next_obs = self.next_obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        weights = np.array(
            [self._calculate_weight(i, beta) for i in indices], dtype=np.float32
        )

        return dict(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
        )

    def _sample_proportional(self) -> List[int]:
        indices = []
        p_total = self.sum_tree.sum(0, self.size)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        p_min = self.min_tree.min(0, self.size) / self.sum_tree.sum(0, self.size)
        max_weight = (p_min * self.size) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum(0, self.size)
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight
        return weight

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            assert priority > 0, "priority must be positive"
            assert 0 <= idx < self.size

            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority**self.alpha
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha


# ─────────────────────────────────────────────
# 3.  NoisyLinear Layer
# ─────────────────────────────────────────────


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )


# ─────────────────────────────────────────────
# 4.  Rainbow Network (Dueling + Noisy + C51)
# ─────────────────────────────────────────────


class RainbowNetwork(nn.Module):
    """
    Combines:
      - Dueling architecture (value + advantage streams)
      - NoisyNet linear layers
      - Distributional output (C51 atoms)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        atom_size: int,
        support: torch.Tensor,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support = support

        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value = NoisyLinear(hidden_dim, atom_size)

        # Advantage stream
        self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage = NoisyLinear(hidden_dim, action_dim * atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the distribution over atoms for each action: [B, action_dim, atom_size]."""
        dist = self.get_distribution(x)
        q = torch.sum(dist * self.support, dim=2)  # [B, action_dim]
        return q

    def get_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Return categorical distributions: [B, action_dim, atom_size]."""
        feature = self.feature(x)

        value = F.relu(self.value_hidden(feature))
        value = self.value(value).view(-1, 1, self.atom_size)  # [B, 1, atom_size]

        advantage = F.relu(self.advantage_hidden(feature))
        advantage = self.advantage(advantage).view(-1, self.action_dim, self.atom_size)

        # Dueling: combine value and advantage
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # avoid log(0) later
        return dist

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        self.value_hidden.reset_noise()
        self.value.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()


# ─────────────────────────────────────────────
# 5.  Rainbow DQN Agent
# ─────────────────────────────────────────────


class RainbowDQNAgent:
    """Rainbow DQN Agent."""

    def __init__(
        self,
        env: gym.Env,
        # Replay
        memory_size: int = 2**14,  # 16384 — must be power of 2
        batch_size: int = 128,
        # Learning
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update_freq: int = 100,
        # PER
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        prior_eps: float = 1e-6,
        # N-step
        n_step: int = 3,
        # C51
        v_min: float = -200.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # Misc
        hidden_dim: int = 128,
        seed: int = 42,
    ):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.prior_eps = prior_eps
        self.n_step = n_step

        # Beta annealing for PER importance-sampling
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # C51 support
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.support = self.support.to(self.device)

        # Replay buffers (1-step PER + n-step)
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha, n_step=n_step, gamma=gamma
        )
        # We also keep a simple 1-step buffer for PER indices (handled within memory)

        # Networks
        self.online_net = RainbowNetwork(
            obs_dim, action_dim, atom_size, self.support, hidden_dim
        ).to(self.device)

        self.target_net = RainbowNetwork(
            obs_dim, action_dim, atom_size, self.support, hidden_dim
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Counters
        self.update_count = 0
        self.transition: list = []

        # Seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def select_action(self, state: np.ndarray) -> int:
        """Select action using noisy network (no ε-greedy needed)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.online_net(state_t)
        return q_values.argmax(dim=1).item()

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition and learn if ready."""
        transition = self.memory.store(state, action, reward, next_state, done)

        if self.memory.size >= self.batch_size:
            loss = self._learn()
            return loss
        return None

    def _anneal_beta(self, frame_idx: int):
        """Linearly anneal beta from beta_start to 1.0."""
        fraction = min(frame_idx / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def _learn(self) -> float:
        """Sample from PER, compute distributional loss, update priorities."""
        samples = self.memory.sample(self.beta)

        weights = torch.FloatTensor(samples["weights"]).to(self.device)
        indices = samples["indices"]

        # Compute distributional loss
        loss_elements = self._compute_loss(samples)
        loss = (loss_elements * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = loss_elements.detach().cpu().numpy() + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # Reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # Target network update
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def _compute_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Categorical distributional RL loss (C51) with Double DQN action selection.
        Returns per-element loss (not reduced) for PER weighting.
        """
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["actions"]).to(self.device)
        reward = torch.FloatTensor(samples["rewards"]).to(self.device)
        done = torch.FloatTensor(samples["dones"]).to(self.device)

        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # ── Double DQN: use online net to select actions ──
            next_action = self.online_net(next_state).argmax(dim=1)  # [B]
            next_dist = self.target_net.get_distribution(next_state)  # [B, A, atoms]
            next_dist = next_dist[range(self.batch_size), next_action]  # [B, atoms]

            # ── Distributional Bellman projection ──
            t_z = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * (
                self.gamma**self.n_step
            ) * self.support.unsqueeze(0)
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / delta_z  # [B, atoms]
            l = b.floor().long()
            u = b.ceil().long()

            # Fix edge case where l == u
            l = l.clamp(0, self.atom_size - 1)
            u = u.clamp(0, self.atom_size - 1)

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # Current distribution
        dist = self.online_net.get_distribution(state)  # [B, A, atoms]
        log_p = torch.log(dist[range(self.batch_size), action])  # [B, atoms]

        # Cross-entropy loss per sample
        element_wise_loss = -(proj_dist * log_p).sum(dim=1)  # [B]

        return element_wise_loss

    def train(self, num_frames: int = 200_000, log_interval: int = 1000):
        """Main training loop."""
        state, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_count = 0
        rewards_history = []
        losses = []

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            loss = self.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            state = next_state

            # Anneal PER beta
            self._anneal_beta(frame_idx)

            if done:
                state, _ = self.env.reset()
                rewards_history.append(episode_reward)
                episode_count += 1
                episode_reward = 0.0

                # Flush remaining n-step transitions
                while len(self.memory.n_step_buffer) > 0:
                    # Store partial n-step returns
                    obs = self.memory.n_step_buffer[0][0]
                    act = self.memory.n_step_buffer[0][1]
                    # Compute partial n-step info
                    r, n_o, d = self.memory._get_n_step_info()
                    idx = self.memory.ptr
                    self.memory.obs[idx] = obs
                    self.memory.next_obs[idx] = n_o
                    self.memory.actions[idx] = act
                    self.memory.rewards[idx] = r
                    self.memory.dones[idx] = d
                    priority_alpha = self.memory.max_priority**self.memory.alpha
                    self.memory.sum_tree[idx] = priority_alpha
                    self.memory.min_tree[idx] = priority_alpha
                    self.memory.ptr = (self.memory.ptr + 1) % self.memory.capacity
                    self.memory.size = min(self.memory.size + 1, self.memory.capacity)
                    self.memory.n_step_buffer.popleft()

            if frame_idx % log_interval == 0:
                avg_reward = np.mean(rewards_history[-50:]) if rewards_history else 0
                avg_loss = np.mean(losses[-200:]) if losses else 0
                print(
                    f"Frame {frame_idx:>7d} | "
                    f"Episodes {episode_count:>4d} | "
                    f"Avg Reward (50 ep): {avg_reward:>8.2f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Beta: {self.beta:.3f}"
                )

        return rewards_history


# ─────────────────────────────────────────────
# 6.  Main — Train on CartPole-v1
# ─────────────────────────────────────────────


def main():
    env = gym.make("CartPole-v1")

    agent = RainbowDQNAgent(
        env=env,
        memory_size=2**14,  # 16384
        batch_size=128,
        gamma=0.99,
        lr=1e-3,
        target_update_freq=150,
        # PER
        alpha=0.6,
        beta_start=0.4,
        beta_frames=50_000,
        prior_eps=1e-6,
        # N-step
        n_step=3,
        # C51
        v_min=0.0,
        v_max=500.0,
        atom_size=51,
        # Architecture
        hidden_dim=128,
        seed=42,
    )

    rewards = agent.train(num_frames=100_000, log_interval=2_000)

    # ── Evaluate ──
    print("\n=== Evaluation (10 episodes, no noise) ===")
    eval_env = gym.make("CartPole-v1", render_mode=None)
    eval_rewards = []
    for ep in range(10):
        state, _ = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            # Use only the mean weights (no noise) for eval
            with torch.no_grad():
                q = agent.online_net(state_t)
            action = q.argmax(dim=1).item()
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        eval_rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.0f}")

    print(
        f"\nMean eval reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}"
    )
    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
