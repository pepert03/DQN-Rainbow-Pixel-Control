import argparse
import torch
from torch import nn
import gymnasium as gym
from dqn import DQN
from buffer import ExperienceReplay
from wrappers import make_env
import itertools
import yaml
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = "./configs/hyperparameters.yml"


class Agent:

    def __init__(self, hyperparameter_set):
        with open(CONFIG, "r") as f:
            all_config = yaml.safe_load(f)
            config = all_config[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id = config["env_id"]
        self.replay_memory_size = config["replay_memory_size"]
        self.mini_batch_size = config["mini_batch_size"]
        self.epsilon_init = config["epsilon_init"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.network_sync_rate = config["network_sync_rate"]
        self.learning_rate = config["learning_rate"]
        self.discount_factor_g = config["discount_factor_g"]

        # Optional MuJoCo env parameters (only used by MuJoCo envs that support them)
        self.env_make_kwargs = {
            k: config[k]
            for k in (
                "forward_reward_weight",
                "ctrl_cost_weight",
            )
            if k in config
        }

        self.optimizer = None
        self.loss_fn = nn.MSELoss()

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")

        # env = gym.make(self.env_id, render_mode="human" if render else None)

        # # DQN needs a discrete action space. If the env is continuous (Box), discretize it.
        # if isinstance(env.action_space, gym.spaces.Box):
        #     env = DiscretizedActionWrapper(env, bins=3)

        env = make_env(self.env_id, render, seed=42, **self.env_make_kwargs)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(state_dim, action_dim).to(device)

        if is_training:
            buffer = ExperienceReplay(capacity=self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(state_dim, action_dim).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Count the number of steps taken (for target network updates)
            step_count = 0

            # Use Adam optimizer for training the DQN
            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate
            )

            best_reward = float("-inf")
        else:
            # Load the trained model for evaluation
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)

            terminated, truncated = False, False
            episode_reward = 0.0

            while not terminated and not truncated:

                if is_training and random.random() < epsilon:
                    # Sample from the action space
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1,2,3,...]) -> tensor([[1,2,3,...]])
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                # Take a step using the sampled action
                next_state, reward, terminated, truncated, info = env.step(
                    action.item()
                )

                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                episode_reward += float(reward)

                if is_training:
                    buffer.push(
                        state, action, reward, next_state, terminated or truncated
                    )
                    step_count += 1

                # Move to new state
                state = next_state

            rewards_per_episode.append(episode_reward)

            if is_training:

                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(buffer) > self.mini_batch_size:
                    mini_batch = buffer.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Stacks
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(next_states).to(device)
        rewards = torch.stack(rewards).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - dones)
                * self.discount_factor_g
                * target_dqn(new_states).max(dim=1)[0]
            )

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
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
