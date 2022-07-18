import numpy as np

from .grid_world import ActionProbs, DPMethod, GridWorld, State, Values


def test_values_dump() -> None:
    values = Values()
    env = GridWorld()
    for state in env.states():
        values.set(state, 1.0)
    values.dump()


def test_dp_one_time() -> None:
    actions = ActionProbs()
    dp = DPMethod()
    values = Values()
    env = GridWorld()

    new_value = dp.policy_eval(actions, values, env, n_iter=1)
    new_value.dump()
    assert np.round(new_value.get(State(1, 3)), 2) == -0.04
    assert np.round(new_value.get(State(0, 3)), 2) == 0.0
    assert np.round(new_value.get(State(0, 2)), 2) == 0.25


def test_dp_full() -> None:
    actions = ActionProbs()
    dp = DPMethod()
    values = Values()
    env = GridWorld()

    new_value = dp.policy_eval(actions, values, env)
    new_value.dump()


def test_dp_actions() -> None:
    dp = DPMethod()
    env = GridWorld()
    action_probs = dp.policy_iter(env)
    action_probs.dump()


def test_dp_complete_code() -> None:
    values = Values()
    env = GridWorld()
    dp = DPMethod()
    values = dp.value_iter(values, env)
    values.dump()
    action_probs = dp.greedy_policy(values, env)
    action_probs.dump()
