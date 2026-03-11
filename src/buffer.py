import random


class ExperienceReplay:
    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        if seed is not None:
            random.seed(seed)

    def push(self, state, action, reward, next_state, done, n_steps=1):
        experience = (state, action, reward, next_state, done, int(n_steps))
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
