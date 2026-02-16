# src/dqn_mujoco.py
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from wrappers import make_env
from buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    
    # --- MUJOCO ENVIRONMENT CONFIG ---
    env_id: str = "dm_control/walker-walk-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning (Reduced to 10k for faster start)"""
    train_frequency: int = 4
    """the frequency of training"""

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Input: 12 channels (4 stacked frames * 3 RGB colors)
        self.network = nn.Sequential(
            nn.Conv2d(12, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        # Ensure input is a float tensor on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        
        # x shape received from Gym: [Batch, Stack, Height, Width, Channel] 
        # Example: [1, 4, 84, 84, 3]
        
        # 1. Reorder dimensions: Move Channel (RGB) before Height and Width
        # From [Batch, Stack, H, W, C] -> [Batch, Stack, C, H, W]
        x = x.permute(0, 1, 4, 2, 3)
        
        # 2. Flatten Stack and Channels (4 * 3 = 12 channels)
        # From [Batch, 4, 3, 84, 84] -> [Batch, 12, 84, 84]
        # This matches what Conv2d expects: (Batch, Channels, Height, Width)
        x = x.flatten(start_dim=1, end_dim=2)
        
        # 3. Normalize pixel values (0-255 -> 0-1)
        return self.network(x / 255.0)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # --- GPU INFO BLOCK ---
    print("-" * 30)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: Using CPU (Training will be slow)")
    print("-" * 30)
    # ----------------------

    # Environment Setup (uses src/wrappers.py)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Buffer Initialization (uses src/buffers.py)
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    
    # Variables for logging
    current_loss = 0.0
    current_q_val = 0.0
    
    # --- MAIN TRAINING LOOP ---
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # Action Selection
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # DIMENSION FIX: using as_tensor allows proper conversion of the numpy array
            q_values = q_network(torch.as_tensor(obs, device=device, dtype=torch.float32))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Logging (Enhanced)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # Fix: Extract item from numpy array to avoid format error
                    ep_return = info['episode']['r'].item()
                    ep_length = info['episode']['l'].item()
                    sps = int(global_step / (time.time() - start_time))
                    
                    print(f"Step={global_step} | Return={ep_return:.2f} | Length={ep_length} | "
                          f"Eps={epsilon:.3f} | Loss={current_loss:.3f} | Q_Val={current_q_val:.1f} | SPS={sps}")
                    
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Handle final observations for vector environments
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Save to buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Optimization Step
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # Store metrics for logging
                current_loss = loss.item()
                current_q_val = old_val.mean().item()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    # Save final model
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    envs.close()
    writer.close()