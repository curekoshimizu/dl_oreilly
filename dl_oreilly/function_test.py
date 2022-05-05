import numpy as np

from . import NDFloatArray
from .function import Exp, Square, add, exp, square
from .variable import Variable


def test_square() -> None:
    x = Variable(np.array([10, 20]))
    f = Square()
    y = f(x)
    assert np.all(y.data == np.array([100, 400]))


def test_call_three_functions() -> None:
    # forward
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
    # forward
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

    z.backward()
    assert np.allclose(w.grad.data, exact(input))


def test_functions() -> None:
    # forward
    x = Variable(np.array([0.5, 0]))
    y = square(exp(square(x)))
    assert np.allclose(y.data, np.array([1.648721270700128, 1]))

    # backward
    def exact(w: NDFloatArray) -> NDFloatArray:
        return np.exp(w**2) ** 2 * w * 4

    y.backward()
    assert np.allclose(x.grad.data, exact(x.data))


def test_add() -> None:
    # forward
    x = Variable(np.array(2))
    y = Variable(np.array(3))
    z = add(x, y)
    assert z.data == 5.0

    # backward
    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0

    # forward
    w = Variable(np.array(1))
    x = Variable(np.array(2))
    y = Variable(np.array(3))
    z = add(w, x, y)
    assert z.data == 6.0

    z = add()
    assert z.data == 0.0
