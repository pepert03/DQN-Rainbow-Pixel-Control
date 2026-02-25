import argparse
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from dqn import Pixel_DQN, DQN
from buffer import ExperienceReplay, PrioritizedExperienceReplay
from wrappers import make_env
import itertools
import yaml
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import time
from typing import Any

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = "./configs/hyperparameters.yml"


class Agent:

    @staticmethod
    def _reset_noisy_layers(model: nn.Module) -> None:
        """Reset noise for NoisyNet layers if present."""
        for module in model.modules():
            if module.__class__.__name__ == "NoisyLinear" and hasattr(
                module, "reset_noise"
            ):
                module.reset_noise()

    def __init__(self, hyperparameter_set):
        # If folder runs/hyperparameter_set exists, we are resuming training, so we load the config.yml from that folder.
        #  Otherwise, we load the config from the main configs folder.
        config_file = os.path.join(RUNS_DIR, hyperparameter_set, "config.yml")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        else:
            with open(CONFIG, "r") as f:
                all_config = yaml.safe_load(f)
                config = all_config[hyperparameter_set]
                # Save the config to the runs folder for future reference
                os.makedirs(os.path.join(RUNS_DIR, hyperparameter_set), exist_ok=True)
                with open(config_file, "w") as f:
                    yaml.dump(config, f)

        self.hyperparameter_set = hyperparameter_set

        self.env_id = config["env_id"]
        self.obs_type = config["obs"]
        self.replay_memory_size = config["replay_memory_size"]
        self.mini_batch_size = config["mini_batch_size"]
        self.epsilon_init = config["epsilon_init"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.network_sync_rate = config["network_sync_rate"]
        self.learning_rate = config["learning_rate"]
        self.discount_factor_g = config["discount_factor_g"]

        # Rainbow DQN
        self.enable_double_dqn = config.get("enable_double_dqn", False)
        self.enable_dueling_dqn = config.get("enable_dueling_dqn", False)
        self.enable_prioritized_replay = config.get("enable_prioritized_replay", False)
        self.enable_noisy_nets = config.get("enable_noisy_nets", False)
        self.enable_distributional = config.get("enable_distributional", False)
        self.enable_n_step = config.get("enable_n_step", False)

        # n-step returns (Rainbow)
        self.n_step = int(config.get("n_step", 3)) if self.enable_n_step else 1
        if self.n_step < 1:
            self.n_step = 1

        # C51 / Distributional DQN
        # NOTE: these keys are optional in configs/hyperparameters.yml
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

        self.optimizer = None
        self.loss_fn = nn.MSELoss()

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "training.log")
        self.MODEL_FILE = os.path.join(
            RUNS_DIR, self.hyperparameter_set, "best_model.pt"
        )
        self.CHECKPOINT_FILE = os.path.join(
            RUNS_DIR, self.hyperparameter_set, "checkpoint.pt"
        )
        self.GRAPH_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "graph.png")

    def _n_step_transition(self, n_step_buffer: deque):
        """Build one n-step transition from the current buffer.

        Returns: (state, action, reward_n, next_state_n, done_n, n_steps_used)
        """
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

    def load_model(self, is_training=True, render=False):

        env = make_env(self.env_id, self.obs_type, render)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Instantiate models up-front (needed for both fresh start and resume)
        if self.obs_type == "pixel":
            policy_dqn = Pixel_DQN(
                state_dim,
                action_dim,
                enable_dueling_dqn=self.enable_dueling_dqn,
                enable_noisy_nets=self.enable_noisy_nets,
                enable_distributional=self.enable_distributional,
                num_atoms=self.num_atoms,
            ).to(device)
            target_dqn = Pixel_DQN(
                state_dim,
                action_dim,
                enable_dueling_dqn=self.enable_dueling_dqn,
                enable_noisy_nets=self.enable_noisy_nets,
                enable_distributional=self.enable_distributional,
                num_atoms=self.num_atoms,
            ).to(device)
        else:
            policy_dqn = DQN(
                state_dim,
                action_dim,
                enable_dueling_dqn=self.enable_dueling_dqn,
                enable_noisy_nets=self.enable_noisy_nets,
                enable_distributional=self.enable_distributional,
                num_atoms=self.num_atoms,
            ).to(device)
            target_dqn = DQN(
                state_dim,
                action_dim,
                enable_dueling_dqn=self.enable_dueling_dqn,
                enable_noisy_nets=self.enable_noisy_nets,
                enable_distributional=self.enable_distributional,
                num_atoms=self.num_atoms,
            ).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Training-only state
        if self.enable_prioritized_replay:
            buffer = PrioritizedExperienceReplay(
                capacity=self.replay_memory_size,
                alpha=self.prioritized_replay_alpha,
                beta=self.prioritized_replay_beta,
            )
        else:
            buffer = ExperienceReplay(capacity=self.replay_memory_size)
        epsilon = self.epsilon_init
        step_count = 0
        if is_training:
            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate
            )

        start_episode = 0
        best_reward = float("-inf")
        rewards_per_episode: list[float] = []
        epsilon_history: list[float] = []

        checkpoint_file = os.path.join(
            RUNS_DIR, self.hyperparameter_set, "checkpoint.pt"
        )

        if is_training:
            if os.path.exists(checkpoint_file):
                checkpoint = torch.load(
                    checkpoint_file, map_location=device, weights_only=False
                )
                policy_dqn.load_state_dict(checkpoint["model_state_dict"])
                target_dqn.load_state_dict(checkpoint["target_model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epsilon = float(checkpoint.get("epsilon", epsilon))
                step_count = int(checkpoint.get("step_count", step_count))
                start_episode = int(checkpoint.get("episode", 0)) + 1
                best_reward = float(checkpoint.get("best_reward", best_reward))
                rewards_per_episode = list(checkpoint.get("rewards_per_episode", []))
                epsilon_history = list(checkpoint.get("epsilon_history", []))

                rb = checkpoint.get("replay_buffer")
                if rb is not None:
                    if self.enable_prioritized_replay:
                        buffer = PrioritizedExperienceReplay(
                            capacity=rb["capacity"],
                            alpha=float(rb.get("alpha", self.prioritized_replay_alpha)),
                            beta=float(rb.get("beta", self.prioritized_replay_beta)),
                        )
                        buffer.buffer = deque(rb["data"], maxlen=rb["capacity"])
                        prios = rb.get("priorities")
                        if prios is None:
                            prios = [1.0 for _ in range(len(buffer.buffer))]
                        buffer.priorities = deque(prios, maxlen=rb["capacity"])
                    else:
                        buffer = ExperienceReplay(capacity=rb["capacity"])
                        buffer.buffer = deque(rb["data"], maxlen=rb["capacity"])

                print(
                    f"Resumed from episode {start_episode-1} | epsilon={epsilon:0.4f} | steps={step_count}"
                )
            else:
                print("Starting training from scratch.")
        else:
            if os.path.exists(self.MODEL_FILE):
                policy_dqn.load_state_dict(
                    torch.load(self.MODEL_FILE, map_location=device)
                )
                target_dqn.load_state_dict(policy_dqn.state_dict())
                policy_dqn.eval()
                print(f"Loaded model weights from: {self.MODEL_FILE}")

        return (
            env,
            policy_dqn,
            target_dqn,
            buffer,
            epsilon,
            rewards_per_episode,
            epsilon_history,
            best_reward,
            start_episode,
            step_count,
        )

    def save_model(
        self,
        policy_dqn,
        target_dqn,
        episode_reward,
        episode,
        best_reward,
        rewards_per_episode,
        epsilon_history,
        buffer,
        epsilon,
        step_count,
    ):

        # Save model if we have a new best reward
        if episode_reward > best_reward:
            log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message + "\n")
            torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

        # Save checkpoint every 100 episodes (no custom-class pickling)
        elif episode % 100 == 0:
            empty_buffer = ExperienceReplay(capacity=self.replay_memory_size)
            replay_state = None
            if self.enable_prioritized_replay:
                replay_state = {
                    "capacity": buffer.capacity,
                    "data": list(buffer.buffer),
                    "priorities": list(buffer.priorities),
                    "alpha": buffer.alpha,
                    "beta": buffer.beta,
                }
            else:
                replay_state = {
                    "capacity": buffer.capacity,
                    "data": list(buffer.buffer),
                }
                
            checkpoint = {
                "model_state_dict": policy_dqn.state_dict(),
                "target_model_state_dict": target_dqn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "replay_buffer": replay_state,
                "epsilon": epsilon,
                "episode": episode,
                "best_reward": best_reward,
                "rewards_per_episode": rewards_per_episode,
                "epsilon_history": epsilon_history,
                "step_count": step_count,
            }
            # Change last checkpint to last_checkpoint for backup, in case saving is interrupted
            if os.path.exists(self.CHECKPOINT_FILE):
                os.replace(self.CHECKPOINT_FILE, self.CHECKPOINT_FILE + ".backup")
            torch.save(checkpoint, self.CHECKPOINT_FILE)
            log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Checkpoint saved at episode {episode}"
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message + "\n")

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")

        # Create environment and load model
        (
            env,
            policy_dqn,
            target_dqn,
            buffer,
            epsilon,
            rewards_per_episode,
            epsilon_history,
            best_reward,
            start_episode,
            step_count,
        ) = self.load_model(is_training=is_training, render=render)

        for episode in itertools.count(start_episode):
            state, _ = env.reset()

            n_step_buffer = deque(maxlen=self.n_step)

            # Keep observations as numpy on CPU to save VRAM.
            terminated, truncated = False, False
            episode_reward = 0.0

            while not terminated and not truncated:

                # With NoisyNets, exploration is handled by the stochastic layers,
                # so we disable epsilon-greedy.
                if (
                    is_training
                    and (not self.enable_noisy_nets)
                    and random.random() < epsilon
                ):
                    # Sample from the action space
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
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
                            action = q_values.squeeze(0).argmax().item()
                        else:
                            action = q_out.squeeze().argmax().item()

                # Take a step using the sampled action
                next_state, reward, terminated, truncated, info = env.step(action)

                # print(info)
                if render:
                    print(
                        f"Episode Reward: {episode_reward:0.1f}, Step Reward: {reward:0.1f}",
                        end="\r",
                    )

                episode_reward += float(reward)

                if is_training:
                    done_flag = terminated or truncated
                    step_count += 1

                    if self.enable_n_step and self.n_step > 1:
                        n_step_buffer.append(
                            (state, action, reward, next_state, done_flag)
                        )

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
                                s0, a0, r_n, ns_n, d_n, steps_used = (
                                    self._n_step_transition(n_step_buffer)
                                )
                                buffer.push(
                                    s0, a0, r_n, ns_n, d_n, n_steps=steps_used
                                )
                                n_step_buffer.popleft()
                    else:
                        buffer.push(state, action, reward, next_state, done_flag, n_steps=1)

                    # Optimize every step once we have enough data
                    if len(buffer) > self.mini_batch_size:
                        if self.enable_prioritized_replay and isinstance(
                            buffer, PrioritizedExperienceReplay
                        ):
                            mini_batch, indices, weights = buffer.sample(
                                self.mini_batch_size
                            )
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

                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                # Move to new state
                state = next_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                self.save_model(
                    policy_dqn,
                    target_dqn,
                    episode_reward,
                    episode,
                    best_reward,
                    rewards_per_episode,
                    epsilon_history,
                    buffer,
                    epsilon,
                    step_count,
                )
                if episode_reward > best_reward:
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

        env.close()

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

        if len(mini_batch[0]) == 6:
            states, actions, rewards, next_states, dones, n_steps = zip(*mini_batch)
        else:
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            n_steps = [1 for _ in range(len(rewards))]

        # Convert CPU numpy -> GPU tensors in a single batch (fast + VRAM-friendly)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        new_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        n_steps_t = torch.tensor(n_steps, dtype=torch.int64, device=device)
        gamma_ns = torch.pow(
            torch.tensor(self.discount_factor_g, dtype=torch.float32, device=device),
            n_steps_t,
        )

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
                w = torch.tensor(weights, dtype=torch.float32, device=device)
                loss = (w * per_sample_loss).mean()
            else:
                loss = per_sample_loss.mean()

            # For PER: priorities from per-sample distributional loss
            td_errors = per_sample_loss.detach()

        else:
            with torch.no_grad():
                if self.enable_double_dqn:
                    # Double DQN: action selection from policy_dqn, value from target_dqn
                    next_actions = policy_dqn(new_states).argmax(dim=1, keepdim=True)
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
                w = torch.tensor(weights, dtype=torch.float32, device=device)
                per_sample_loss = td_errors.pow(2)
                loss = (w * per_sample_loss).mean()
            else:
                loss = self.loss_fn(current_q, target_q)

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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])
        plt.subplot(121)
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)
        plt.subplot(122)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")

    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
