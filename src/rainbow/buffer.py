from collections import deque
import random

from src.buffer import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, capacity, alpha=0.6, beta=0.4, seed=None):
        super().__init__(capacity, seed)
        self.alpha = alpha  # Tunes the degree of prioritization (0 - no prioritization, 1 - full prioritization)
        self.beta = beta  # Tunes the degree of importance-sampling correction (0 - no correction, 1 - full correction)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, n_steps=1):
        super().push(state, action, reward, next_state, done, n_steps=n_steps)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        priorities = list(self.priorities)
        probabilities = [p**self.alpha for p in priorities]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        indices = random.choices(
            range(len(self.buffer)), weights=probabilities, k=batch_size
        )
        samples = [self.buffer[i] for i in indices]

        # Calculate importance-sampling weights
        weights = [
            (len(self.buffer) * probabilities[i]) ** (-self.beta) for i in indices
        ]
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
