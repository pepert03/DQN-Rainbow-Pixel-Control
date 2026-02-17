import torch
from torch import nn
import torch.nn.functional as F


# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        # Ensure input is a float tensor on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(
                x, dtype=torch.float32, device=next(self.parameters()).device
            )

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


if __name__ == "__main__":
    state_dim = 17  # Example state dimension for HalfCheetah-v5
    action_dim = 6  # Example action dimension for HalfCheetah-v5
    model = DQN(state_dim, action_dim)
    print(model)
