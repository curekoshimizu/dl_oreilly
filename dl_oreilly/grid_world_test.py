from .grid_world import GridWorld, Values


def test_values_dump() -> None:
    values = Values()
    env = GridWorld()
    for state in env.states():
        values.update(state, 1.0)
    values.dump()
