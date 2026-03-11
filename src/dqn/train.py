from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import numpy as np
from datetime import datetime, timedelta
import time
from torch.utils.tensorboard import SummaryWriter

from ..config import load_config, get_paths, device, DATE_FORMAT
from ..networks import DQN, Pixel_DQN
from ..buffer import ExperienceReplay
from ..wrappers import make_env, make_vec_env
from ..utils import (
    load_checkpoint,
    load_best_model,
    save_best_model,
    save_checkpoint,
    save_graph,
)


loss_fn = nn.MSELoss()


def build_models(config, env):
    """Build policy and target DQN models (vanilla, no Rainbow features)."""
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    act_space = getattr(env, "single_action_space", env.action_space)
    state_dim = obs_space.shape[0]
    action_dim = act_space.n

    if config["obs"] == "pixel":
        policy_dqn = Pixel_DQN(state_dim, action_dim).to(device)
        target_dqn = Pixel_DQN(state_dim, action_dim).to(device)
    else:
        policy_dqn = DQN(state_dim, action_dim).to(device)
        target_dqn = DQN(state_dim, action_dim).to(device)

    target_dqn.load_state_dict(policy_dqn.state_dict())
    return policy_dqn, target_dqn


def optimize(mini_batch, policy_dqn, target_dqn, optimizer, discount_factor_g):
    """Standard DQN optimization step."""
    if len(mini_batch[0]) == 6:
        states, actions, rewards, next_states, dones, n_steps = zip(*mini_batch)
    else:
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        n_steps = [1 for _ in range(len(rewards))]

    # Convert CPU numpy -> GPU tensors in a single batch (fast + VRAM-friendly)
    states = torch.from_numpy(np.stack(states)).float().to(device)
    actions = torch.as_tensor(actions, dtype=torch.int64, device=device)
    new_states = torch.from_numpy(np.stack(next_states)).float().to(device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    n_steps_t = torch.as_tensor(n_steps, dtype=torch.int64, device=device)
    gamma_ns = torch.pow(
        torch.tensor(discount_factor_g, dtype=torch.float32, device=device),
        n_steps_t,
    )

    with torch.no_grad():
        target_q = (
            rewards + (1 - dones) * gamma_ns * target_dqn(new_states).max(dim=1)[0]
        )

    current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

    loss = loss_fn(current_q, target_q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(hyperparameter_set):
    """Train a vanilla DQN agent with vectorized environments."""
    config = load_config(hyperparameter_set)
    paths = get_paths(hyperparameter_set)

    num_envs = config.get("num_envs", 30)
    envs = make_vec_env(config["env_id"], config["obs"], num_envs=num_envs)
    policy_dqn, target_dqn = build_models(config, envs)
    buffer = ExperienceReplay(capacity=config["replay_memory_size"])
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=config["learning_rate"])

    train_frequency = config.get("train_frequency", 4)

    epsilon = config["epsilon_init"]
    step_count = 0
    episode_count = 0
    best_reward = float("-inf")
    rewards_per_episode = []
    epsilon_history = []
    last_loss = None

    # Try to load checkpoint
    checkpoint_data = load_checkpoint(
        paths["checkpoint"], policy_dqn, target_dqn, optimizer, epsilon
    )
    if checkpoint_data is not None:
        (
            epsilon,
            step_count,
            episode_count,
            best_reward,
            rewards_per_episode,
            epsilon_history,
        ) = checkpoint_data

    # TensorBoard
    writer = SummaryWriter(paths["tensorboard"])

    start_time = datetime.now()
    last_graph_update_time = start_time

    log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
    print(log_message)
    with open(paths["log"], "w") as file:
        file.write(log_message + "\n")

    iter_timer = time.perf_counter()
    iter_count = 0

    # Vectorized env state
    episode_rewards = np.zeros(num_envs)
    states, _ = envs.reset()
    action_dim = envs.single_action_space.n

    for global_step in itertools.count():
        # Per-env epsilon-greedy
        with torch.no_grad():
            states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
            q_actions = policy_dqn(states_t).argmax(dim=1).cpu().numpy()

        random_actions = np.random.randint(0, action_dim, size=num_envs)
        explore_mask = np.random.random(num_envs) < epsilon
        actions = np.where(explore_mask, random_actions, q_actions)

        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        episode_rewards += rewards
        step_count += 1

        # Store transitions for each sub-env
        for i in range(num_envs):
            # Gymnasium auto-resets: next_states[i] is reset obs for done envs
            # Use final_observation for the true terminal observation
            if (
                dones[i]
                and "final_observation" in infos
                and infos["_final_observation"][i]
            ):
                next_obs = infos["final_observation"][i]
            else:
                next_obs = next_states[i]
            buffer.push(
                states[i],
                int(actions[i]),
                float(rewards[i]),
                next_obs,
                bool(dones[i]),
                n_steps=1,
            )

        # Handle completed episodes
        for i in range(num_envs):
            if dones[i]:
                ep_reward = float(episode_rewards[i])
                rewards_per_episode.append(ep_reward)
                episode_count += 1
                iter_count += 1

                # TensorBoard: episode metrics
                writer.add_scalar("episode/reward", ep_reward, episode_count)
                mean_100 = float(np.mean(rewards_per_episode[-100:]))
                writer.add_scalar("episode/mean_reward_100", mean_100, episode_count)
                writer.add_scalar("episode/best_reward", best_reward, episode_count)

                if ep_reward > best_reward:
                    save_best_model(
                        policy_dqn,
                        paths["model"],
                        paths["log"],
                        ep_reward,
                        best_reward,
                        episode_count,
                    )
                    best_reward = ep_reward
                elif episode_count % 100 == 0:
                    save_checkpoint(
                        policy_dqn,
                        optimizer,
                        epsilon,
                        episode_count,
                        best_reward,
                        rewards_per_episode,
                        epsilon_history,
                        step_count,
                        paths["checkpoint"],
                        paths["log"],
                        last_loss=last_loss,
                    )

                episode_rewards[i] = 0.0

        # Optimize every train_frequency steps once we have enough data
        if (
            len(buffer) > config["mini_batch_size"]
            and step_count % train_frequency == 0
        ):
            mini_batch = buffer.sample(config["mini_batch_size"])
            last_loss = optimize(
                mini_batch,
                policy_dqn,
                target_dqn,
                optimizer,
                config["discount_factor_g"],
            )

            # TensorBoard: training metrics
            writer.add_scalar("train/loss", last_loss, global_step)
            writer.add_scalar("train/epsilon", epsilon, global_step)
            writer.add_scalar("train/buffer_size", len(buffer), global_step)

            epsilon = max(epsilon * config["epsilon_decay"], config["epsilon_min"])
            epsilon_history.append(epsilon)

            if step_count > config["network_sync_rate"]:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

        states = next_states

        # Print iterations per second
        elapsed = time.perf_counter() - iter_timer
        if elapsed >= 1.0 and iter_count > 0:
            it_per_sec = iter_count / elapsed
            mean_reward = np.mean(rewards_per_episode[-100:])
            loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
            print(
                f"Episode {episode_count} | {it_per_sec:.1f} ep/s | "
                f"Reward: {rewards_per_episode[-1]:.1f} | Mean(100): {mean_reward:.1f} | "
                f"Loss: {loss_str} | Epsilon: {epsilon:.4f}"
            )
            writer.add_scalar("perf/episodes_per_second", it_per_sec, episode_count)
            iter_timer = time.perf_counter()
            iter_count = 0

        # Update graph every 10 seconds
        current_time = datetime.now()
        if current_time - last_graph_update_time > timedelta(seconds=10):
            save_graph(rewards_per_episode, epsilon_history, paths["graph"])
            last_graph_update_time = current_time

    writer.close()
    envs.close()


def evaluate(hyperparameter_set, render=True):
    """Evaluate a vanilla DQN agent."""
    config = load_config(hyperparameter_set)
    paths = get_paths(hyperparameter_set)

    env = make_env(config["env_id"], config["obs"], render=render)
    policy_dqn, target_dqn = build_models(config, env)
    load_best_model(paths["model"], policy_dqn, target_dqn)

    for episode in itertools.count():
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0.0

        while not terminated and not truncated:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = policy_dqn(state_tensor).squeeze().argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)

            if render:
                print(
                    f"Episode Reward: {episode_reward:0.1f}, Step Reward: {reward:0.1f}",
                    end="\r",
                )

            episode_reward += float(reward)
            state = next_state

    env.close()
