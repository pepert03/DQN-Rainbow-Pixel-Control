import random

from ..buffer import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, capacity, alpha=0.6, beta=0.4, seed=None):
        super().__init__(capacity, seed)
        self.alpha = alpha  # Tunes the degree of prioritization (0 - no prioritization, 1 - full prioritization)
        self.beta = beta  # Tunes the degree of importance-sampling correction (0 - no correction, 1 - full correction)
        self.priorities = []
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done, n_steps=1):
        if len(self.buffer) < self.capacity:
            self.priorities.append(self._max_priority)
        else:
            self.priorities[self.position] = self._max_priority
        super().push(state, action, reward, next_state, done, n_steps=n_steps)

    def sample(self, batch_size):
        n = len(self.buffer)
        probabilities = [p**self.alpha for p in self.priorities[:n]]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        indices = random.choices(range(n), weights=probabilities, k=batch_size)
        samples = [self.buffer[i] for i in indices]

        # Calculate importance-sampling weights
        weights = [(n * probabilities[i]) ** (-self.beta) for i in indices]
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            if priority > self._max_priority:
                self._max_priority = priority
