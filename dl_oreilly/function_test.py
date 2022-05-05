import numpy as np

from . import NDFloatArray
from .function import Exp, Square
from .variable import Variable


def test_square() -> None:
    x = Variable(np.array([10, 20]))
    f = Square()
    y = f(x)
    assert np.all(y.data == np.array([100, 400]))


def test_call_three_functions() -> None:
    # foward
    f = Square()
    g = Exp()
    h = Square()

    input = np.array([0.5, 0])
    w = Variable(input)
    x = f(w)
    y = g(x)
    z = h(y)
    assert np.allclose(z.data, np.array([1.648721270700128, 1]))

    # backward
    def exact(w: NDFloatArray) -> NDFloatArray:
        return np.exp(w**2) ** 2 * w * 4

    grad_w = np.array([1.0])
    grad_x = h.backward(grad_w)
    grad_y = g.backward(grad_x)
    grad_z = f.backward(grad_y)
    assert np.allclose(grad_z.data, exact(input))


def test_backward_function() -> None:
    # foward
    f = Square()
    g = Exp()
    h = Square()

    input = np.array([0.5, 0])
    w = Variable(input)
    x = f(w)
    y = g(x)
    z = h(y)
    assert np.allclose(z.data, np.array([1.648721270700128, 1]))

    # backward
    def exact(w: NDFloatArray) -> NDFloatArray:
        return np.exp(w**2) ** 2 * w * 4

    grad_w = z.backward()
    assert np.allclose(grad_w.data, exact(input))
