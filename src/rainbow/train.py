from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Any

from src.dqn.train import DQNAgent
from src.rainbow.buffer import PrioritizedExperienceReplay
from src.config import device


class RainbowAgent(DQNAgent):
    """Rainbow DQN agent — extends DQNAgent with:
    - Prioritized Experience Replay
    - NoisyNets
    - Distributional RL (C51)
    - N-step returns
    """

    def __init__(self, hyperparameter_set):
        super().__init__(hyperparameter_set)
        config = self.config

        # Enable Rainbow flags from config
        self.enable_prioritized_replay = config.get("enable_prioritized_replay", False)
        self.enable_noisy_nets = config.get("enable_noisy_nets", False)
        self.enable_distributional = config.get("enable_distributional", False)
        self.enable_n_step = config.get("enable_n_step", False)

        # n-step returns
        self.n_step = int(config.get("n_step", 3)) if self.enable_n_step else 1
        if self.n_step < 1:
            self.n_step = 1

        # C51 / Distributional DQN
        self.num_atoms = int(config.get("num_atoms", config.get("atom_size", 51)))
        self.v_min = float(config.get("v_min", -200.0))
        self.v_max = float(config.get("v_max", 200.0))
        if self.enable_distributional:
            if self.v_max <= self.v_min:
                raise ValueError("distributional v_max must be > v_min")
            if self.num_atoms < 2:
                raise ValueError("distributional num_atoms must be >= 2")
            self.support = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=device
            )
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Prioritized replay
        self.prioritized_replay_alpha = float(
            config.get("prioritized_replay_alpha", 0.6)
        )
        self.prioritized_replay_beta = float(config.get("prioritized_replay_beta", 0.4))
        self.prioritized_replay_beta_increment = float(
            config.get("prioritized_replay_beta_increment", 0.0)
        )
        self.prioritized_replay_eps = float(config.get("prioritized_replay_eps", 1e-6))

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _create_buffer(self):
        if self.enable_prioritized_replay:
            return PrioritizedExperienceReplay(
                capacity=self.replay_memory_size,
                alpha=self.prioritized_replay_alpha,
                beta=self.prioritized_replay_beta,
            )
        return super()._create_buffer()

    def _select_action(self, state, policy_dqn, epsilon, is_training, env):
        # With NoisyNets, exploration is handled by the stochastic layers,
        # so we disable epsilon-greedy.
        if is_training and (not self.enable_noisy_nets) and random.random() < epsilon:
            return env.action_space.sample()
        else:
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", enabled=(device.type == "cuda")),
            ):
                if is_training and self.enable_noisy_nets:
                    self._reset_noisy_layers(policy_dqn)
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_out = policy_dqn(state_tensor)
                if self.enable_distributional:
                    # q_out: [1, A, atoms] logits
                    probs = F.softmax(q_out, dim=-1)
                    q_values = (probs * self.support).sum(dim=-1)  # [1, A]
                    return q_values.squeeze(0).argmax().item()
                else:
                    return q_out.squeeze().argmax().item()

    def _select_actions_batch(self, states, policy_dqn, epsilon, num_envs, action_dim):
        """Batched action selection with NoisyNets + distributional support."""
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=(device.type == "cuda")),
        ):
            if self.enable_noisy_nets:
                self._reset_noisy_layers(policy_dqn)
            state_tensor = torch.tensor(
                np.array(states), dtype=torch.float32, device=device
            )
            q_out = policy_dqn(state_tensor)
            if self.enable_distributional:
                probs = F.softmax(q_out, dim=-1)
                q_values = (probs * self.support).sum(dim=-1)
                greedy_actions = q_values.argmax(dim=-1).cpu().numpy()
            else:
                greedy_actions = q_out.argmax(dim=-1).cpu().numpy()

        if not self.enable_noisy_nets:
            explore = np.random.random(num_envs) < epsilon
            random_actions = np.random.randint(0, action_dim, size=num_envs)
            return np.where(explore, random_actions, greedy_actions)
        return greedy_actions

    def _n_step_transition(self, n_step_buffer: deque):
        """Build one n-step transition from the current buffer."""
        reward_n = 0.0
        next_state_n = n_step_buffer[-1][3]
        done_n = False
        steps_used = 0

        for i, (_, _, r, ns, d) in enumerate(n_step_buffer):
            reward_n += (self.discount_factor_g**i) * float(r)
            next_state_n = ns
            steps_used = i + 1
            if d:
                done_n = True
                break

        s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1]
        return s0, a0, reward_n, next_state_n, done_n, steps_used

    def _store_transition(
        self, buffer, state, action, reward, next_state, done_flag, n_step_buffer
    ):
        if self.enable_n_step and self.n_step > 1:
            n_step_buffer.append((state, action, reward, next_state, done_flag))

            # When we have enough transitions, push one aggregated n-step experience
            if len(n_step_buffer) >= self.n_step:
                s0, a0, r_n, ns_n, d_n, steps_used = self._n_step_transition(
                    n_step_buffer
                )
                buffer.push(s0, a0, r_n, ns_n, d_n, n_steps=steps_used)
                n_step_buffer.popleft()

            # If episode ended, flush remaining partial n-step transitions
            if done_flag:
                while n_step_buffer:
                    s0, a0, r_n, ns_n, d_n, steps_used = self._n_step_transition(
                        n_step_buffer
                    )
                    buffer.push(s0, a0, r_n, ns_n, d_n, n_steps=steps_used)
                    n_step_buffer.popleft()
        else:
            buffer.push(state, action, reward, next_state, done_flag, n_steps=1)

    def _sample_and_optimize(self, buffer, policy_dqn, target_dqn):
        if len(buffer) > self.mini_batch_size:
            if self.enable_prioritized_replay and isinstance(
                buffer, PrioritizedExperienceReplay
            ):
                mini_batch, indices, weights = buffer.sample(self.mini_batch_size)
                self.optimize(
                    mini_batch,
                    policy_dqn,
                    target_dqn,
                    buffer=buffer,
                    indices=indices,
                    weights=weights,
                )
            else:
                mini_batch = buffer.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
            return True
        return False

    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn,
        buffer: Any | None = None,
        indices: list[int] | None = None,
        weights: list[float] | None = None,
    ):

        if self.enable_noisy_nets:
            # Fresh noise per gradient step.
            self._reset_noisy_layers(policy_dqn)
            self._reset_noisy_layers(target_dqn)

        states, actions, rewards, next_states, dones, n_steps = mini_batch

        # numpy arrays → GPU tensors
        states = torch.as_tensor(states).to(device, dtype=torch.float32)
        actions = torch.as_tensor(actions).to(device, dtype=torch.int64)
        new_states = torch.as_tensor(next_states).to(device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards).to(device, dtype=torch.float32)
        dones = torch.as_tensor(dones).to(device, dtype=torch.float32)
        n_steps_t = torch.as_tensor(n_steps).to(device, dtype=torch.int64)
        gamma_ns = torch.pow(
            torch.tensor(self.discount_factor_g, dtype=torch.float32, device=device),
            n_steps_t,
        )

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if self.enable_distributional:
                # C51 categorical distributional RL loss.
                # policy_dqn(states): [B, A, atoms] logits
                logits = policy_dqn(states)
                log_probs = F.log_softmax(logits, dim=-1)

                # Select action-specific distributions
                actions_idx = actions.view(-1, 1, 1).expand(-1, 1, self.num_atoms)
                log_probs_a = log_probs.gather(1, actions_idx).squeeze(1)  # [B, atoms]

                with torch.no_grad():
                    # Next action selection (Double DQN uses policy net for argmax)
                    if self.enable_double_dqn:
                        next_logits_policy = policy_dqn(new_states)
                        next_probs_policy = F.softmax(next_logits_policy, dim=-1)
                        next_q_policy = (next_probs_policy * self.support).sum(dim=-1)
                        next_actions = next_q_policy.argmax(dim=1)
                    else:
                        next_logits_target = target_dqn(new_states)
                        next_probs_target = F.softmax(next_logits_target, dim=-1)
                        next_q_target = (next_probs_target * self.support).sum(dim=-1)
                        next_actions = next_q_target.argmax(dim=1)

                    next_logits = target_dqn(new_states)
                    next_probs = F.softmax(next_logits, dim=-1)
                    next_actions_idx = next_actions.view(-1, 1, 1).expand(
                        -1, 1, self.num_atoms
                    )
                    next_dist = next_probs.gather(1, next_actions_idx).squeeze(
                        1
                    )  # [B, atoms]

                    # ── Distributional Bellman projection ──
                    t_z = rewards.unsqueeze(1) + (1 - dones).unsqueeze(
                        1
                    ) * gamma_ns.unsqueeze(1) * self.support.unsqueeze(0)
                    t_z = t_z.clamp(self.v_min, self.v_max)
                    b = (t_z - self.v_min) / self.delta_z
                    l = b.floor().long()
                    u = b.ceil().long()

                    m = torch.zeros_like(next_dist)
                    batch_size = rewards.shape[0]
                    offset = (
                        torch.arange(batch_size, device=device).unsqueeze(1)
                        * self.num_atoms
                    )

                    l_idx = (l + offset).view(-1)
                    u_idx = (u + offset).view(-1)
                    m_flat = m.view(-1)
                    m_flat.index_add_(
                        0,
                        l_idx,
                        (next_dist * (u.float() - b)).view(-1),
                    )
                    m_flat.index_add_(
                        0,
                        u_idx,
                        (next_dist * (b - l.float())).view(-1),
                    )

                per_sample_loss = -(m * log_probs_a).sum(dim=1)  # cross-entropy
                if weights is not None:
                    w = torch.as_tensor(weights).to(device, dtype=torch.float32)
                    loss = (w * per_sample_loss).mean()
                else:
                    loss = per_sample_loss.mean()

                # For PER: priorities from per-sample distributional loss
                td_errors = per_sample_loss.detach()

            else:
                with torch.no_grad():
                    if self.enable_double_dqn:
                        # Double DQN: action selection from policy_dqn, value from target_dqn
                        next_actions = policy_dqn(new_states).argmax(
                            dim=1, keepdim=True
                        )
                        target_q = (
                            rewards
                            + (1 - dones)
                            * gamma_ns
                            * target_dqn(new_states).gather(1, next_actions).squeeze()
                        )
                    else:
                        target_q = (
                            rewards
                            + (1 - dones)
                            * gamma_ns
                            * target_dqn(new_states).max(dim=1)[0]
                        )

                current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
                td_errors = current_q - target_q

                if weights is not None:
                    w = torch.as_tensor(weights).to(device, dtype=torch.float32)
                    per_sample_loss = td_errors.pow(2)
                    loss = (w * per_sample_loss).mean()
                else:
                    loss = self.loss_fn(current_q, target_q.float())

        # Update PER priorities from absolute TD error
        if (
            buffer is not None
            and indices is not None
            and self.enable_prioritized_replay
            and isinstance(buffer, PrioritizedExperienceReplay)
        ):
            new_priorities = (
                td_errors.detach().abs().cpu().numpy() + self.prioritized_replay_eps
            ).tolist()
            buffer.update_priorities(indices, new_priorities)
            if self.prioritized_replay_beta_increment > 0.0:
                buffer.beta = min(
                    1.0, buffer.beta + self.prioritized_replay_beta_increment
                )

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
