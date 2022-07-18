import numpy as np

from .grid_world import Actions, DPMethod, GridWorld, State, Values


def test_values_dump() -> None:
    values = Values()
    env = GridWorld()
    for state in env.states():
        values.set(state, 1.0)
    values.dump()


def test_dp_one_time() -> None:
    actions = Actions()
    dp = DPMethod()
    values = Values()
    env = GridWorld()

    new_value = dp.policy_eval(actions, values, env, n_iter=1)
    new_value.dump()
    assert np.round(new_value.get(State(1, 3)), 2) == -0.04
    assert np.round(new_value.get(State(0, 3)), 2) == 0.0
    assert np.round(new_value.get(State(0, 2)), 2) == 0.25


def test_dp_full() -> None:
    actions = Actions()
    dp = DPMethod()
    values = Values()
    env = GridWorld()

    new_value = dp.policy_eval(actions, values, env)
    new_value.dump()
