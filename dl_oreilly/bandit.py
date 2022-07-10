import numpy as np


class Bandit:
    def __init__(self, n_arms: int = 10) -> None:
        self._rates = np.random.rand(n_arms)

    def play(self, arm_index: int) -> int:
        rate = self._rates[arm_index]
        return int(rate > np.random.rand())


class Agent:
    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self._epsilon = epsilon
        self._qs = np.zeros(action_size)
        self._counts = np.zeros(action_size)

    def update(self, action: int, reward: float) -> None:
        self._counts[action] += 1
        self._qs[action] += (reward - self._qs[action]) / self._counts[action]

    def get_action(self) -> int:
        """
        e-greed method
        """
        if np.random.rand() < self._epsilon:
            # choose action randomly
            return int(np.random.randint(0, len(self._qs)))
        return int(np.argmax(self._qs))
