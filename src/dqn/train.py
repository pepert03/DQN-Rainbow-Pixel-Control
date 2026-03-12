from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
from src.networks import Pixel_DQN, DQN
from src.buffer import ExperienceReplay
from src.wrappers import make_env, make_vec_env
from src.config import load_config, device, RUNS_DIR, DATE_FORMAT
from src.utils import save_graph
import itertools
import random
import numpy as np
import os
import time
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


class DQNAgent:

    @staticmethod
    def _reset_noisy_layers(model: nn.Module) -> None:
        """Reset noise for NoisyNet layers if present."""
        for module in model.modules():
            if module.__class__.__name__ == "NoisyLinear" and hasattr(
                module, "reset_noise"
            ):
                module.reset_noise()

    def __init__(self, hyperparameter_set):
        config = load_config(hyperparameter_set)
        self.config = config
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
        self.train_frequency = int(config.get("train_frequency", 4))
        self.num_envs = int(config.get("num_envs", 1))

        # DQN extensions (can be enabled independently)
        self.enable_double_dqn = config.get("enable_double_dqn", False)
        self.enable_dueling_dqn = config.get("enable_dueling_dqn", False)

        # Rainbow flags — disabled in vanilla DQN
        self.enable_prioritized_replay = False
        self.enable_noisy_nets = False
        self.enable_distributional = False
        self.enable_n_step = False
        self.n_step = 1
        self.num_atoms = 51

        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        self._amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "training.log")
        self.MODEL_FILE = os.path.join(
            RUNS_DIR, self.hyperparameter_set, "best_model.pt"
        )
        self.CHECKPOINT_FILE = os.path.join(
            RUNS_DIR, self.hyperparameter_set, "checkpoint.pt"
        )
        self.GRAPH_FILE = os.path.join(RUNS_DIR, self.hyperparameter_set, "graph.png")
        self.TB_DIR = os.path.join(RUNS_DIR, self.hyperparameter_set, "tensorboard")

    # ------------------------------------------------------------------
    # Overridable helpers (Rainbow overrides these)
    # ------------------------------------------------------------------

    def _create_buffer(self):
        return ExperienceReplay(capacity=self.replay_memory_size)

    def _create_network(self, obs_dim, action_dim):
        if self.obs_type == "pixel":
            return Pixel_DQN(
                obs_dim,  # full obs_shape tuple for pixel
                action_dim,
                enable_dueling_dqn=self.enable_dueling_dqn,
                enable_noisy_nets=self.enable_noisy_nets,
                enable_distributional=self.enable_distributional,
                num_atoms=self.num_atoms,
            )
        else:
            return DQN(
                obs_dim,  # scalar state_dim for MLP
                action_dim,
                enable_dueling_dqn=self.enable_dueling_dqn,
                enable_noisy_nets=self.enable_noisy_nets,
                enable_distributional=self.enable_distributional,
                num_atoms=self.num_atoms,
            )

    def _select_action(self, state, policy_dqn, epsilon, is_training, env):
        if is_training and random.random() < epsilon:
            return env.action_space.sample()
        else:
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", enabled=(device.type == "cuda")),
            ):
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                return policy_dqn(state_tensor).squeeze().argmax().item()

    def _select_actions_batch(self, states, policy_dqn, epsilon, num_envs, action_dim):
        """Select actions for a batch of states (vectorized envs)."""
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=(device.type == "cuda")),
        ):
            state_tensor = torch.tensor(
                np.array(states), dtype=torch.float32, device=device
            )
            q_values = policy_dqn(state_tensor)
            greedy_actions = q_values.argmax(dim=-1).cpu().numpy()

        explore = np.random.random(num_envs) < epsilon
        random_actions = np.random.randint(0, action_dim, size=num_envs)
        return np.where(explore, random_actions, greedy_actions)

    def _store_transition(
        self, buffer, state, action, reward, next_state, done_flag, n_step_buffer
    ):
        buffer.push(state, action, reward, next_state, done_flag, n_steps=1)

    def _sample_and_optimize(self, buffer, policy_dqn, target_dqn):
        if len(buffer) > self.mini_batch_size:
            mini_batch = buffer.sample(self.mini_batch_size)
            self.optimize(mini_batch, policy_dqn, target_dqn)
            return True
        return False

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def load_model(self, is_training=True, render=False):

        env = make_env(self.env_id, self.obs_type, render)

        obs_shape = env.observation_space.shape
        obs_dim = obs_shape if self.obs_type == "pixel" else obs_shape[0]
        action_dim = env.action_space.n

        policy_dqn = self._create_network(obs_dim, action_dim).to(device)
        target_dqn = self._create_network(obs_dim, action_dim).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        buffer = self._create_buffer()
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
                if "target_model_state_dict" in checkpoint:
                    target_dqn.load_state_dict(checkpoint["target_model_state_dict"])
                else:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epsilon = float(checkpoint.get("epsilon", epsilon))
                step_count = int(checkpoint.get("step_count", step_count))
                start_episode = int(checkpoint.get("episode", 0)) + 1
                best_reward = float(checkpoint.get("best_reward", best_reward))
                rewards_per_episode = list(checkpoint.get("rewards_per_episode", []))
                epsilon_history = list(checkpoint.get("epsilon_history", []))

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
            checkpoint = {
                "model_state_dict": policy_dqn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": epsilon,
                "episode": episode,
                "best_reward": best_reward,
                "rewards_per_episode": rewards_per_episode,
                "epsilon_history": epsilon_history,
                "step_count": step_count,
            }
            if os.path.exists(self.CHECKPOINT_FILE):
                os.remove(self.CHECKPOINT_FILE)
            backup_file = self.CHECKPOINT_FILE + ".backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            torch.save(checkpoint, self.CHECKPOINT_FILE)
            log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Checkpoint saved at episode {episode}"
            print(log_message)
            with open(self.LOG_FILE, "a") as file:
                file.write(log_message + "\n")

    def run(self, is_training=True, render=False):
        if is_training and self.num_envs > 1:
            return self._run_vectorized()
        return self._run_single(is_training, render)

    def _run_vectorized(self):
        """Training loop with vectorized (parallel) environments."""
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting ({self.num_envs} envs)..."
        print(log_message)
        with open(self.LOG_FILE, "w") as file:
            file.write(log_message + "\n")

        writer = SummaryWriter(log_dir=self.TB_DIR)

        # Create vectorized env
        envs = make_vec_env(self.env_id, self.obs_type, self.num_envs)
        obs_shape = envs.single_observation_space.shape
        obs_dim = obs_shape if self.obs_type == "pixel" else obs_shape[0]
        action_dim = envs.single_action_space.n

        policy_dqn = self._create_network(obs_dim, action_dim).to(device)
        target_dqn = self._create_network(obs_dim, action_dim).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        buffer = self._create_buffer()
        epsilon = self.epsilon_init
        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=self.learning_rate
        )

        rewards_per_episode: list[float] = []
        epsilon_history: list[float] = []
        best_reward = float("-inf")
        episode_count = 0
        step_count = 0
        steps_since_train = 0
        steps_since_sync = 0

        # Resume from checkpoint
        checkpoint_file = os.path.join(
            RUNS_DIR, self.hyperparameter_set, "checkpoint.pt"
        )
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(
                checkpoint_file, map_location=device, weights_only=False
            )
            policy_dqn.load_state_dict(checkpoint["model_state_dict"])
            if "target_model_state_dict" in checkpoint:
                target_dqn.load_state_dict(checkpoint["target_model_state_dict"])
            else:
                target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epsilon = float(checkpoint.get("epsilon", epsilon))
            step_count = int(checkpoint.get("step_count", 0))
            episode_count = int(checkpoint.get("episode", 0)) + 1
            best_reward = float(checkpoint.get("best_reward", best_reward))
            rewards_per_episode = list(checkpoint.get("rewards_per_episode", []))
            epsilon_history = list(checkpoint.get("epsilon_history", []))
            print(
                f"Resumed from episode {episode_count - 1} | epsilon={epsilon:0.4f} | steps={step_count}"
            )
        else:
            print("Starting training from scratch.")

        # Per-env tracking
        n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.num_envs)]
        ep_rewards = np.zeros(self.num_envs)
        ep_steps = np.zeros(self.num_envs, dtype=int)

        # Periodic printing
        last_print_time = time.time()
        last_print_steps = step_count

        states, _ = envs.reset()

        try:
            while True:
                actions = self._select_actions_batch(
                    states, policy_dqn, epsilon, self.num_envs, action_dim
                )
                next_states, rewards, terminateds, truncateds, infos = envs.step(
                    actions
                )
                dones = np.logical_or(terminateds, truncateds)

                ep_rewards += rewards
                ep_steps += 1
                step_count += self.num_envs
                steps_since_train += self.num_envs
                steps_since_sync += self.num_envs

                # Store transitions per env
                final_obs = infos.get("final_observation", [None] * self.num_envs)
                for i in range(self.num_envs):
                    next_s = (
                        final_obs[i]
                        if dones[i] and final_obs[i] is not None
                        else next_states[i]
                    )
                    self._store_transition(
                        buffer,
                        states[i],
                        int(actions[i]),
                        float(rewards[i]),
                        next_s,
                        bool(dones[i]),
                        n_step_buffers[i],
                    )

                # Handle completed episodes (silent — no print per episode)
                for i in range(self.num_envs):
                    if dones[i]:
                        ep_reward = float(ep_rewards[i])
                        rewards_per_episode.append(ep_reward)
                        mean_reward = np.mean(
                            rewards_per_episode[
                                max(0, len(rewards_per_episode) - 100) :
                            ]
                        )

                        writer.add_scalar("reward/episode", ep_reward, episode_count)
                        writer.add_scalar("reward/mean_100", mean_reward, episode_count)
                        writer.add_scalar(
                            "reward/best",
                            max(best_reward, ep_reward),
                            episode_count,
                        )
                        writer.add_scalar("training/epsilon", epsilon, episode_count)
                        writer.add_scalar(
                            "training/episode_steps",
                            ep_steps[i],
                            episode_count,
                        )
                        writer.add_scalar(
                            "training/total_steps", step_count, episode_count
                        )
                        writer.add_scalar(
                            "training/buffer_size",
                            len(buffer),
                            episode_count,
                        )

                        self.save_model(
                            policy_dqn,
                            target_dqn,
                            ep_reward,
                            episode_count,
                            best_reward,
                            rewards_per_episode,
                            epsilon_history,
                            epsilon,
                            step_count,
                        )
                        if ep_reward > best_reward:
                            best_reward = ep_reward

                        ep_rewards[i] = 0.0
                        ep_steps[i] = 0
                        n_step_buffers[i] = deque(maxlen=self.n_step)
                        episode_count += 1

                # Print summary every ~1 second
                now = time.time()
                if now - last_print_time >= 1.0:
                    elapsed = now - last_print_time
                    sps = (step_count - last_print_steps) / elapsed
                    mean_reward = (
                        np.mean(
                            rewards_per_episode[
                                max(0, len(rewards_per_episode) - 100) :
                            ]
                        )
                        if rewards_per_episode
                        else 0.0
                    )
                    writer.add_scalar("training/steps_per_second", sps, step_count)
                    print(
                        f"Episode {episode_count} | Reward: {rewards_per_episode[-1] if rewards_per_episode else 0:.1f}"
                        f" | Mean100: {mean_reward:.1f}"
                        f" | Epsilon: {epsilon:.4f}"
                        f" | Steps: {step_count}"
                        f" | SPS: {sps:.0f}"
                    )
                    last_print_time = now
                    last_print_steps = step_count

                # Optimize
                if steps_since_train >= self.train_frequency:
                    if self._sample_and_optimize(buffer, policy_dqn, target_dqn):
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)
                    steps_since_train = 0

                # Sync target network
                if steps_since_sync >= self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps_since_sync = 0

                # Update graph periodically
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    save_graph(self.GRAPH_FILE, rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                states = next_states
        finally:
            writer.close()
            envs.close()

    def _run_single(self, is_training=True, render=False):
        writer = None
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")

            writer = SummaryWriter(log_dir=self.TB_DIR)

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
            episode_steps = 0
            episode_start_time = time.time()

            while not terminated and not truncated:

                action = self._select_action(
                    state, policy_dqn, epsilon, is_training, env
                )

                # Take a step using the sampled action
                next_state, reward, terminated, truncated, info = env.step(action)

                if render:
                    print(
                        f"Episode Reward: {episode_reward:0.1f}, Step Reward: {reward:0.1f}",
                        end="\r",
                    )

                episode_reward += float(reward)
                episode_steps += 1

                if is_training:
                    done_flag = terminated or truncated
                    step_count += 1

                    self._store_transition(
                        buffer,
                        state,
                        action,
                        reward,
                        next_state,
                        done_flag,
                        n_step_buffer,
                    )

                    if (
                        step_count % self.train_frequency == 0
                        and self._sample_and_optimize(buffer, policy_dqn, target_dqn)
                    ):
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                # Move to new state
                state = next_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                # TensorBoard logging
                episode_elapsed = time.time() - episode_start_time
                steps_per_sec = (
                    episode_steps / episode_elapsed if episode_elapsed > 0 else 0.0
                )
                print(
                    f"Episode {episode} | Reward: {episode_reward:0.1f} | Mean100: {np.mean(rewards_per_episode[-100:]):0.1f} | Epsilon: {epsilon:0.4f} | Steps: {episode_steps} | Steps/sec: {steps_per_sec:0.2f}"
                )
                mean_reward = np.mean(
                    rewards_per_episode[max(0, len(rewards_per_episode) - 100) :]
                )

                writer.add_scalar("reward/episode", episode_reward, episode)
                writer.add_scalar("reward/mean_100", mean_reward, episode)
                writer.add_scalar(
                    "reward/best",
                    best_reward if episode_reward <= best_reward else episode_reward,
                    episode,
                )
                writer.add_scalar("training/epsilon", epsilon, episode)
                writer.add_scalar("training/steps_per_second", steps_per_sec, episode)
                writer.add_scalar("training/episode_steps", episode_steps, episode)
                writer.add_scalar("training/total_steps", step_count, episode)
                writer.add_scalar("training/buffer_size", len(buffer), episode)

                self.save_model(
                    policy_dqn,
                    target_dqn,
                    episode_reward,
                    episode,
                    best_reward,
                    rewards_per_episode,
                    epsilon_history,
                    epsilon,
                    step_count,
                )
                if episode_reward > best_reward:
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    save_graph(self.GRAPH_FILE, rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

        if writer is not None:
            writer.close()
        env.close()

    # Alias for backward compatibility
    _run = _run_single

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, rewards, next_states, dones, _ = mini_batch

        # numpy arrays → GPU tensors (uint8 obs converted to float32 on GPU)
        states = torch.as_tensor(states).to(device, dtype=torch.float32)
        actions = torch.as_tensor(actions).to(device, dtype=torch.int64)
        new_states = torch.as_tensor(next_states).to(device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards).to(device, dtype=torch.float32)
        dones = torch.as_tensor(dones).to(device, dtype=torch.float32)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            with torch.no_grad():
                if self.enable_double_dqn:
                    next_actions = policy_dqn(new_states).argmax(dim=1, keepdim=True)
                    target_q = (
                        rewards
                        + (1 - dones)
                        * self.discount_factor_g
                        * target_dqn(new_states).gather(1, next_actions).squeeze()
                    )
                else:
                    target_q = (
                        rewards
                        + (1 - dones)
                        * self.discount_factor_g
                        * target_dqn(new_states).max(dim=1)[0]
                    )

            current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = self.loss_fn(current_q, target_q.float())

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
