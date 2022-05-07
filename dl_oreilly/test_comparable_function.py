from .variable import ComparableFunction, DummyFunction


def test_comparable_function() -> None:
    f = DummyFunction(1)
    g = DummyFunction(2)
    h = DummyFunction(2)

    x = ComparableFunction(f)
    y = ComparableFunction(g)
    z = ComparableFunction(h)

    assert x < y
    assert x < z
    assert y != z
    assert x == x
    assert y == y
    assert z == z
