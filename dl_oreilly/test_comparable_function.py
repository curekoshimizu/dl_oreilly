from .variable import DummyFunction, _ComparableFunction, _FunctionPriorityQueue


def test_comparable_function() -> None:
    f = DummyFunction(1)
    g = DummyFunction(2)
    h = DummyFunction(2)

    x = _ComparableFunction(f)
    y = _ComparableFunction(g)
    z = _ComparableFunction(h)

    assert x > y
    assert x > z
    assert y == z
    assert x == x
    assert y == y
    assert z == z


def test_function_priority_queue() -> None:
    f = DummyFunction(1)
    g = DummyFunction(2)
    h = DummyFunction(2)

    queue = _FunctionPriorityQueue()
    assert queue.is_empty()
    assert queue.register(f)
    assert not queue.is_empty()
    assert queue.register(g)

    assert queue.pop() == g
    assert not queue.register(g)
    assert queue.register(h)

    assert queue.pop() == h
    assert queue.pop() == f

    assert queue.is_empty()
