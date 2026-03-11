import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .config import DATE_FORMAT, device


def load_checkpoint(checkpoint_path, policy_dqn, target_dqn, optimizer, epsilon_init):
    """Load training checkpoint if it exists.

    Returns: (epsilon, step_count, start_episode, best_reward, rewards_per_episode, epsilon_history)
    or None if no checkpoint found.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        policy_dqn.load_state_dict(checkpoint["model_state_dict"])
        if "target_model_state_dict" in checkpoint:
            target_dqn.load_state_dict(checkpoint["target_model_state_dict"])
        else:
            target_dqn.load_state_dict(policy_dqn.state_dict())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epsilon = float(checkpoint.get("epsilon", epsilon_init))
        step_count = int(checkpoint.get("step_count", 0))
        start_episode = int(checkpoint.get("episode", 0)) + 1
        best_reward = float(checkpoint.get("best_reward", float("-inf")))
        rewards_per_episode = list(checkpoint.get("rewards_per_episode", []))
        epsilon_history = list(checkpoint.get("epsilon_history", []))

        print(
            f"Resumed from episode {start_episode-1} | epsilon={epsilon:0.4f} | steps={step_count}"
        )
        return (
            epsilon,
            step_count,
            start_episode,
            best_reward,
            rewards_per_episode,
            epsilon_history,
        )
    else:
        print("Starting training from scratch.")
        return None


def load_best_model(model_path, policy_dqn, target_dqn):
    """Load best model weights for evaluation."""
    if os.path.exists(model_path):
        policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
        target_dqn.load_state_dict(policy_dqn.state_dict())
        policy_dqn.eval()
        print(f"Loaded model weights from: {model_path}")


def save_best_model(
    policy_dqn, model_path, log_path, episode_reward, best_reward, episode
):
    """Save model when a new best reward is achieved."""
    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
    print(log_message)
    with open(log_path, "a") as file:
        file.write(log_message + "\n")
    torch.save(policy_dqn.state_dict(), model_path)


def save_checkpoint(
    policy_dqn,
    optimizer,
    epsilon,
    episode,
    best_reward,
    rewards_per_episode,
    epsilon_history,
    step_count,
    checkpoint_path,
    log_path,
    last_loss=None,
):
    """Save training checkpoint every N episodes."""
    checkpoint = {
        "model_state_dict": policy_dqn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": epsilon,
        "episode": episode,
        "best_reward": best_reward,
        "rewards_per_episode": rewards_per_episode,
        "epsilon_history": epsilon_history,
        "step_count": step_count,
        "last_loss": last_loss,
    }
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    backup_file = checkpoint_path + ".backup"
    if os.path.exists(backup_file):
        os.remove(backup_file)
    torch.save(checkpoint, checkpoint_path)
    log_message = (
        f"{datetime.now().strftime(DATE_FORMAT)}: Checkpoint saved at episode {episode}"
    )
    print(log_message)
    with open(log_path, "a") as file:
        file.write(log_message + "\n")


def save_graph(rewards_per_episode, epsilon_history, graph_path):
    """Save training progress graph."""
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
    fig.savefig(graph_path)
    plt.close(fig)
