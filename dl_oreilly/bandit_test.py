import numpy as np

from .bandit import Agent, AlphaAgent, Bandit, NonBandit


def test_bandit(
    seed: int = 0,
    steps: int = 1000,
    n_arms: int = 10,
) -> None:
    np.random.seed(seed)

    total_reward = 0.0
    bandit = Bandit(n_arms)
    agent = Agent(epsilon=0.1, action_size=n_arms)
    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
    assert total_reward == 896.0


def test_non_bandit(
    seed: int = 0,
    steps: int = 1000,
    n_arms: int = 10,
) -> None:
    np.random.seed(seed)

    total_reward = 0.0
    bandit = NonBandit(n_arms)
    agent = AlphaAgent(alpha=0.8, epsilon=0.1, action_size=n_arms)
    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
    assert total_reward == 942.0
