import numpy as np

from .function import Square
from .variable import Variable


def test_square() -> None:
    x = Variable(np.array([10, 20]))
    f = Square()
    y = f(x)
    assert np.all(y.data == np.array([100, 400]))
