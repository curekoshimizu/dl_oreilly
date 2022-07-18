from .grid_world import Actions, DPMethod, GridWorld, Values


def test_values_dump() -> None:
    values = Values()
    env = GridWorld()
    for state in env.states():
        values.set(state, 1.0)
    values.dump()


def test_dp() -> None:
    actions = Actions()
    dp = DPMethod()
    values = Values()
    env = GridWorld()

    new_value = dp.policy_eval(actions, values, env)
    new_value.dump()
