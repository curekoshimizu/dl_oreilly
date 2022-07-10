import numpy as np

from .bandit import Bandit


def test_bandit(seed: int = 0) -> None:
    np.random.seed(seed)

    N = 10
    q = 0.0
    bandit = Bandit()
    for n in range(1, N + 1):
        reward = bandit.play(0)
        q += (reward - q) / n
