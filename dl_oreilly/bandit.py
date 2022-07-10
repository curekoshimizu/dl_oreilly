import numpy as np


class Bandit:
    def __init__(self, narms: int = 10) -> None:
        self._rates = np.random.rand(narms)

    def play(self, arm_index: int) -> float:
        rate = self._rates[arm_index]
        if rate > np.random.rand():
            return 1.0
        else:
            return 0.0
