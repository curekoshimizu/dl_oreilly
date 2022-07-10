import numpy as np

from .bandit import Agent, Bandit


def test_bandit(seed: int = 0) -> None:
    np.random.seed(seed)

    steps = 1000
    total_reward = 0.0
    epsilon = 0.1
    n_arms = 10
    bandit = Bandit(n_arms)
    agent = Agent(epsilon, n_arms)
    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
    assert total_reward == 896.0
