import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    state_dim = 17  # Example state dimension for HalfCheetah-v5
    action_dim = 6  # Example action dimension for HalfCheetah-v5
    model = DQN(state_dim, action_dim)
    print(model)
