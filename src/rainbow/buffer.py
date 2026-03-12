import numpy as np

from src.buffer import ExperienceReplay


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, capacity, alpha=0.6, beta=0.4, seed=None):
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta = beta
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done, n_steps=1):
        idx = self._idx  # save before parent increments
        super().push(state, action, reward, next_state, done, n_steps=n_steps)
        self._priorities[idx] = self._max_priority

    def sample(self, batch_size):
        probs = self._priorities[: self._size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self._size, size=batch_size, p=probs)

        samples = (
            self._obs[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices],
            self._n_steps[indices],
        )

        weights = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        priorities_arr = np.asarray(priorities, dtype=np.float64)
        self._priorities[indices] = priorities_arr
        self._max_priority = max(self._max_priority, float(priorities_arr.max()))
