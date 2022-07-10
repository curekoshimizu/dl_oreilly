import numpy as np


class Bandit:
    """
    定常問題 (Stationary Problem) : 報酬の確率分布が一定
    """

    def __init__(self, n_arms: int = 10) -> None:
        self._n_arms = n_arms
        self._rates = np.random.rand(n_arms)

    def play(self, arm_index: int) -> int:
        rate = self._rates[arm_index]
        return int(rate > np.random.rand())


class NonBandit(Bandit):
    """
    非定常問題 (Non-Stationary Problem)
    """

    def play(self, arm_index: int) -> int:
        self._rates += 0.1 * np.random.randn(self._n_arms)  # add noise
        return super().play(arm_index)


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


class AlphaAgent(Agent):
    def __init__(self, alpha: float, epsilon: float, action_size: int = 10) -> None:
        super().__init__(epsilon, action_size)
        self._alpha = alpha

    def update(self, action: int, reward: float) -> None:
        self._counts[action] += 1
        self._qs[action] += (reward - self._qs[action]) * self._alpha
