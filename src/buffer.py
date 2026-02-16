from collections import deque
import random


class ExperienceReplay:
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
