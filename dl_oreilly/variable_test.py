import numpy as np

from .variable import Var


def test_variable() -> None:
    x = Var(np.array([1.0, 2.0]), name="x")
    assert x.ndim == 1
    assert x.size == 2
    assert x.shape == (2,)
    assert x.dtype == np.dtype(np.float64)
    assert len(x) == 2
    assert repr(x) == "variable(x:[1. 2.])"


def test_operation() -> None:
    x = Var(np.array(3.0), name="x")
    y = Var(np.array(2.0), name="y")
    z = Var(np.array(1.0), name="y")

    w = x * y + z
    w.backward()
    assert w.data == 7.0
    assert x.grad == 2.0
    assert y.grad == 3.0
