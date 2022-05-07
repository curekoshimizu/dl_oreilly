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


def test_operation_1() -> None:
    x = Var(np.array(3.0), name="x")
    y = Var(np.array(2.0), name="y")
    z = Var(np.array(1.0), name="z")

    w = x * y + z
    w.backward()
    assert w.data == 7.0
    assert x.grad == 2.0
    assert y.grad == 3.0


def test_operation_2() -> None:
    x = Var(np.array(3.0), name="x")
    y = 2.0
    z = 1.0

    w = x * y + z
    w.backward()
    assert w.data == 7.0
    assert x.grad == 2.0


def test_operation_3() -> None:
    x = Var(np.array(3.0), name="x")
    y = 2.0
    z = 1.0

    w = z + y * x
    w.backward()
    assert w.data == 7.0
    assert x.grad == 2.0


def test_operation_neg() -> None:
    x = Var(np.array(2.0))
    assert (-x).data == -2.0


def test_operation_add() -> None:
    x = Var(np.array(2.0))
    assert (2.0 + x).data == 4.0
    assert (x + 4.0).data == 6.0


def test_operation_sub() -> None:
    x = Var(np.array(2.0))
    assert (1.0 - x).data == -1.0
    assert (x - 1.0).data == 1.0


def test_operation_mul() -> None:
    x = Var(np.array(2.0))
    assert (2.0 * x).data == 4.0
    assert (x * 4.0).data == 8.0


def test_operation_div() -> None:
    x = Var(np.array(2.0))
    assert (1.0 / x).data == 0.5
    assert (x / 4.0).data == 0.5


def test_operation_pow() -> None:
    x = Var(np.array(2.0))
    y = x**3
    y.backward()
    assert y.data == 8.0
    assert x.grad == 12.0
