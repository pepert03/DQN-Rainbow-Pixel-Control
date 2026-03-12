import numpy as np


class ExperienceReplay:
    """Pre-allocated numpy replay buffer. Avoids per-sample tuple/list overhead."""

    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self._obs = None  # lazy-init on first push
        self._next_obs = None
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.uint8)
        self._n_steps = np.ones(capacity, dtype=np.int32)
        self._idx = 0
        self._size = 0

        if seed is not None:
            np.random.seed(seed)

    def push(self, state, action, reward, next_state, done, n_steps=1):
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        if self._obs is None:
            self._obs = np.zeros((self.capacity, *state.shape), dtype=state.dtype)
            self._next_obs = np.zeros(
                (self.capacity, *next_state.shape), dtype=next_state.dtype
            )
        self._obs[self._idx] = state
        self._next_obs[self._idx] = next_state
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = done
        self._n_steps[self._idx] = n_steps
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._obs[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices],
            self._n_steps[indices],
        )

    def __len__(self):
        return self._size
