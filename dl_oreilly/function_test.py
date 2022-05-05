import numpy as np

from .function import Exp, Square
from .variable import Variable


def test_square() -> None:
    x = Variable(np.array([10, 20]))
    f = Square()
    y = f(x)
    assert np.all(y.data == np.array([100, 400]))


def test_call_three_functions() -> None:
    f = Square()
    g = Exp()
    h = Square()

    x = Variable(np.array([0.5, 0]))
    y = h(g(f(x)))
    assert np.allclose(y.data, np.array([1.648721270700128, 1]))
