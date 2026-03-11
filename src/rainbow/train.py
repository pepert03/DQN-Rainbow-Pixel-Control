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
from .buffer import PrioritizedExperienceReplay
from ..wrappers import make_env, make_vec_env
from ..utils import (
    load_checkpoint,
    load_best_model,
    save_best_model,
    save_checkpoint,
    save_graph,
)


loss_fn = nn.MSELoss()


def reset_noisy_layers(model):
    """Reset noise for NoisyNet layers if present."""
    for module in model.modules():
        if module.__class__.__name__ == "NoisyLinear" and hasattr(
            module, "reset_noise"
        ):
            module.reset_noise()


def n_step_transition(n_step_buffer, discount_factor_g):
    """Build one n-step transition from the current buffer.

    Returns: (state, action, reward_n, next_state_n, done_n, n_steps_used)
    """
    reward_n = 0.0
    next_state_n = n_step_buffer[-1][3]
    done_n = False
    steps_used = 0

    for i, (_, _, r, ns, d) in enumerate(n_step_buffer):
        reward_n += (discount_factor_g**i) * float(r)
        next_state_n = ns
        steps_used = i + 1
        if d:
            done_n = True
            break

    s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1]
    return s0, a0, reward_n, next_state_n, done_n, steps_used


def build_models(config, env):
    """Build policy and target DQN models with Rainbow features."""
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    act_space = getattr(env, "single_action_space", env.action_space)
    state_dim = obs_space.shape[0]
    action_dim = act_space.n

    enable_dueling = config.get("enable_dueling_dqn", False)
    enable_noisy = config.get("enable_noisy_nets", False)
    enable_dist = config.get("enable_distributional", False)
    num_atoms = int(config.get("num_atoms", config.get("atom_size", 51)))

    if config["obs"] == "pixel":
        policy_dqn = Pixel_DQN(
            state_dim,
            action_dim,
            enable_dueling_dqn=enable_dueling,
            enable_noisy_nets=enable_noisy,
            enable_distributional=enable_dist,
            num_atoms=num_atoms,
        ).to(device)
        target_dqn = Pixel_DQN(
            state_dim,
            action_dim,
            enable_dueling_dqn=enable_dueling,
            enable_noisy_nets=enable_noisy,
            enable_distributional=enable_dist,
            num_atoms=num_atoms,
        ).to(device)
    else:
        policy_dqn = DQN(
            state_dim,
            action_dim,
            enable_dueling_dqn=enable_dueling,
            enable_noisy_nets=enable_noisy,
            enable_distributional=enable_dist,
            num_atoms=num_atoms,
        ).to(device)
        target_dqn = DQN(
            state_dim,
            action_dim,
            enable_dueling_dqn=enable_dueling,
            enable_noisy_nets=enable_noisy,
            enable_distributional=enable_dist,
            num_atoms=num_atoms,
        ).to(device)

    target_dqn.load_state_dict(policy_dqn.state_dict())
    return policy_dqn, target_dqn


def optimize(
    mini_batch,
    policy_dqn,
    target_dqn,
    optimizer,
    config,
    *,
    support=None,
    delta_z=None,
    buffer=None,
    indices=None,
    weights=None,
):
    """Rainbow DQN optimization step (handles distributional, double, PER)."""
    enable_noisy_nets = config.get("enable_noisy_nets", False)
    enable_distributional = config.get("enable_distributional", False)
    enable_double_dqn = config.get("enable_double_dqn", False)
    enable_prioritized_replay = config.get("enable_prioritized_replay", False)
    discount_factor_g = config["discount_factor_g"]
    num_atoms = int(config.get("num_atoms", config.get("atom_size", 51)))
    v_min = float(config.get("v_min", -200.0))
    v_max = float(config.get("v_max", 200.0))
    prioritized_replay_eps = float(config.get("prioritized_replay_eps", 1e-6))
    prioritized_replay_beta_increment = float(
        config.get("prioritized_replay_beta_increment", 0.0)
    )

    if enable_noisy_nets:
        # Fresh noise per gradient step.
        reset_noisy_layers(policy_dqn)
        reset_noisy_layers(target_dqn)

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

    if enable_distributional:
        # C51 categorical distributional RL loss.
        # policy_dqn(states): [B, A, atoms] logits
        logits = policy_dqn(states)
        log_probs = F.log_softmax(logits, dim=-1)

        # Select action-specific distributions
        actions_idx = actions.view(-1, 1, 1).expand(-1, 1, num_atoms)
        log_probs_a = log_probs.gather(1, actions_idx).squeeze(1)  # [B, atoms]

        with torch.no_grad():
            # Next action selection (Double DQN uses policy net for argmax)
            if enable_double_dqn:
                next_logits_policy = policy_dqn(new_states)
                next_probs_policy = F.softmax(next_logits_policy, dim=-1)
                next_q_policy = (next_probs_policy * support).sum(dim=-1)
                next_actions = next_q_policy.argmax(dim=1)
            else:
                next_logits_target = target_dqn(new_states)
                next_probs_target = F.softmax(next_logits_target, dim=-1)
                next_q_target = (next_probs_target * support).sum(dim=-1)
                next_actions = next_q_target.argmax(dim=1)

            next_logits = target_dqn(new_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions_idx = next_actions.view(-1, 1, 1).expand(-1, 1, num_atoms)
            next_dist = next_probs.gather(1, next_actions_idx).squeeze(1)  # [B, atoms]

            # ── Distributional Bellman projection ──
            t_z = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * gamma_ns.unsqueeze(
                1
            ) * support.unsqueeze(0)
            t_z = t_z.clamp(v_min, v_max)
            b = (t_z - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros_like(next_dist)
            batch_size = rewards.shape[0]
            offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_atoms

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
            if enable_double_dqn:
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
                    + (1 - dones) * gamma_ns * target_dqn(new_states).max(dim=1)[0]
                )

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        td_errors = current_q - target_q

        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32, device=device)
            per_sample_loss = td_errors.pow(2)
            loss = (w * per_sample_loss).mean()
        else:
            loss = loss_fn(current_q, target_q)

    # Update PER priorities from absolute TD error
    if (
        buffer is not None
        and indices is not None
        and enable_prioritized_replay
        and isinstance(buffer, PrioritizedExperienceReplay)
    ):
        new_priorities = (
            td_errors.detach().abs().cpu().numpy() + prioritized_replay_eps
        ).tolist()
        buffer.update_priorities(indices, new_priorities)
        if prioritized_replay_beta_increment > 0.0:
            buffer.beta = min(1.0, buffer.beta + prioritized_replay_beta_increment)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(hyperparameter_set):
    """Train a Rainbow DQN agent with vectorized environments."""
    config = load_config(hyperparameter_set)
    paths = get_paths(hyperparameter_set)

    enable_noisy_nets = config.get("enable_noisy_nets", False)
    enable_distributional = config.get("enable_distributional", False)
    enable_prioritized_replay = config.get("enable_prioritized_replay", False)
    enable_n_step = config.get("enable_n_step", False)
    discount_factor_g = config["discount_factor_g"]

    # n-step returns (Rainbow)
    n_step = int(config.get("n_step", 3)) if enable_n_step else 1
    if n_step < 1:
        n_step = 1

    # C51 / Distributional DQN
    num_atoms = int(config.get("num_atoms", config.get("atom_size", 51)))
    v_min = float(config.get("v_min", -200.0))
    v_max = float(config.get("v_max", 200.0))
    support = None
    delta_z = None
    if enable_distributional:
        if v_max <= v_min:
            raise ValueError("distributional v_max must be > v_min")
        if num_atoms < 2:
            raise ValueError("distributional num_atoms must be >= 2")
        support = torch.linspace(v_min, v_max, num_atoms, device=device)
        delta_z = (v_max - v_min) / (num_atoms - 1)

    # Prioritized replay
    prioritized_replay_alpha = float(config.get("prioritized_replay_alpha", 0.6))
    prioritized_replay_beta = float(config.get("prioritized_replay_beta", 0.4))

    num_envs = config.get("num_envs", 30)
    envs = make_vec_env(config["env_id"], config["obs"], num_envs=num_envs)
    policy_dqn, target_dqn = build_models(config, envs)

    if enable_prioritized_replay:
        buffer = PrioritizedExperienceReplay(
            capacity=config["replay_memory_size"],
            alpha=prioritized_replay_alpha,
            beta=prioritized_replay_beta,
        )
    else:
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

    # Per-env n-step buffers
    n_step_bufs = [deque(maxlen=n_step) for _ in range(num_envs)]

    for global_step in itertools.count():
        # Action selection for all envs
        if (not enable_noisy_nets) and np.random.random() < epsilon:
            # All envs explore (fast path)
            actions = np.random.randint(0, action_dim, size=num_envs)
        else:
            with torch.no_grad():
                if enable_noisy_nets:
                    reset_noisy_layers(policy_dqn)
                states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                q_out = policy_dqn(states_t)
                if enable_distributional:
                    probs = F.softmax(q_out, dim=-1)
                    q_values = (probs * support).sum(dim=-1)
                    greedy_actions = q_values.argmax(dim=1).cpu().numpy()
                else:
                    greedy_actions = q_out.argmax(dim=1).cpu().numpy()

            if not enable_noisy_nets:
                # Per-env epsilon-greedy
                random_actions = np.random.randint(0, action_dim, size=num_envs)
                explore_mask = np.random.random(num_envs) < epsilon
                actions = np.where(explore_mask, random_actions, greedy_actions)
            else:
                actions = greedy_actions

        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        episode_rewards += rewards
        step_count += 1

        # Store transitions for each sub-env
        for i in range(num_envs):
            if (
                dones[i]
                and "final_observation" in infos
                and infos["_final_observation"][i]
            ):
                next_obs = infos["final_observation"][i]
            else:
                next_obs = next_states[i]

            if enable_n_step and n_step > 1:
                n_step_bufs[i].append(
                    (
                        states[i],
                        int(actions[i]),
                        float(rewards[i]),
                        next_obs,
                        bool(dones[i]),
                    )
                )

                if len(n_step_bufs[i]) >= n_step:
                    s0, a0, r_n, ns_n, d_n, steps_used = n_step_transition(
                        n_step_bufs[i], discount_factor_g
                    )
                    buffer.push(s0, a0, r_n, ns_n, d_n, n_steps=steps_used)

                if dones[i]:
                    while n_step_bufs[i]:
                        s0, a0, r_n, ns_n, d_n, steps_used = n_step_transition(
                            n_step_bufs[i], discount_factor_g
                        )
                        buffer.push(s0, a0, r_n, ns_n, d_n, n_steps=steps_used)
                        n_step_bufs[i].popleft()
            else:
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
                n_step_bufs[i].clear()

        # Optimize every train_frequency steps
        if (
            len(buffer) > config["mini_batch_size"]
            and step_count % train_frequency == 0
        ):
            if enable_prioritized_replay and isinstance(
                buffer, PrioritizedExperienceReplay
            ):
                mini_batch, indices, weights = buffer.sample(config["mini_batch_size"])
                last_loss = optimize(
                    mini_batch,
                    policy_dqn,
                    target_dqn,
                    optimizer,
                    config,
                    support=support,
                    delta_z=delta_z,
                    buffer=buffer,
                    indices=indices,
                    weights=weights,
                )
            else:
                mini_batch = buffer.sample(config["mini_batch_size"])
                last_loss = optimize(
                    mini_batch,
                    policy_dqn,
                    target_dqn,
                    optimizer,
                    config,
                    support=support,
                    delta_z=delta_z,
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

        # Print speed
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
    """Evaluate a Rainbow DQN agent."""
    config = load_config(hyperparameter_set)
    paths = get_paths(hyperparameter_set)

    enable_distributional = config.get("enable_distributional", False)
    num_atoms = int(config.get("num_atoms", config.get("atom_size", 51)))
    v_min = float(config.get("v_min", -200.0))
    v_max = float(config.get("v_max", 200.0))
    support = None
    if enable_distributional:
        support = torch.linspace(v_min, v_max, num_atoms, device=device)

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
                q_out = policy_dqn(state_tensor)
                if enable_distributional:
                    # q_out: [1, A, atoms] logits
                    probs = F.softmax(q_out, dim=-1)
                    q_values = (probs * support).sum(dim=-1)  # [1, A]
                    action = q_values.squeeze(0).argmax().item()
                else:
                    action = q_out.squeeze().argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)

            if render:
                print(
                    f"Episode Reward: {episode_reward:0.1f}, Step Reward: {reward:0.1f}",
                    end="\r",
                )

            episode_reward += float(reward)
            state = next_state

    env.close()
