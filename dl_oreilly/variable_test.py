import numpy as np

from .protocol import Variable
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
    x = Var(np.array([3.0]), name="x")
    y = Var(np.array([2.0]), name="y")
    z = Var(np.array([1.0]), name="z")

    w = x * y + z
    w.backward()
    assert w.data == 7.0
    assert x.grad.data == 2.0
    assert y.grad.data == 3.0


def test_operation_2() -> None:
    x = Var(np.array([3.0]), name="x")
    y = 2.0
    z = 1.0

    w = x * y + z
    w.backward()
    assert w.data == 7.0
    assert x.grad.data == 2.0


def test_operation_3() -> None:
    x = Var(np.array([3.0]), name="x")
    y = 2.0
    z = 1.0

    w = z + y * x
    w.backward()
    assert w.data == 7.0
    assert x.grad.data == 2.0


def test_operation_neg() -> None:
    x = Var(np.array([2.0]))
    assert (-x).data == -2.0


def test_operation_add() -> None:
    x = Var(np.array([2.0]))
    assert (2.0 + x).data == 4.0
    assert (x + 4.0).data == 6.0


def test_operation_sub() -> None:
    x = Var(np.array([2.0]))
    assert (1.0 - x).data == -1.0
    assert (x - 1.0).data == 1.0


def test_operation_mul() -> None:
    x = Var(np.array([2.0]))
    assert (2.0 * x).data == 4.0
    assert (x * 4.0).data == 8.0


def test_operation_div() -> None:
    x = Var(np.array([2.0]))
    assert (1.0 / x).data == 0.5
    assert (x / 4.0).data == 0.5


def test_operation_pow() -> None:
    x = Var(np.array([2.0]))
    y = x**3
    y.backward()
    assert y.data == 8.0
    assert x.grad.data == 12.0


def test_sphere() -> None:
    def sphere(x: Variable, y: Variable) -> Variable:
        return x**2 + y**2

    x = Var(np.array([1.0]))
    y = Var(np.array([1.0]))
    z = sphere(x, y)
    z.backward()
    assert x.grad.data == 2.0
    assert y.grad.data == 2.0


def test_matyas() -> None:
    def matyas(x: Variable, y: Variable) -> Variable:
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    x = Var(np.array([1.0]))
    y = Var(np.array([1.0]))
    z = matyas(x, y)
    z.backward()
    assert x.grad.data == 0.040000000000000036
    assert y.grad.data == 0.040000000000000036


def test_goldstein_price() -> None:
    def f(x: Variable, y: Variable) -> Variable:
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y * y)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y * y)
        )

    x = Var(np.array([1.0]))
    y = Var(np.array([1.0]))
    z = f(x, y)
    z.backward()
    assert x.grad.data == -5376.0
    assert y.grad.data == 8064.0
