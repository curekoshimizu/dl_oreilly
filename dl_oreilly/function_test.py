import numpy as np

from . import NDFloatArray
from .function import (
    Exp,
    Square,
    add,
    broadcast_to,
    cos,
    diff_f,
    exp,
    matmul,
    mul,
    reshape,
    sin,
    square,
    sum,
    sum_to,
    tanh,
    transpose,
)
from .protocol import Variable
from .variable import Var


def test_square() -> None:
    x = Var(np.array([10, 20]))
    f = Square()
    y = f(x)
    assert np.all(y.data == np.array([100, 400]))


def test_call_three_functions() -> None:
    # forward
    f = Square()
    g = Exp()
    h = Square()

    input = np.array([0.5, 0])
    w = Var(input)
    x = f(w)
    y = g(x)
    z = h(y)
    assert np.allclose(z.data, np.array([1.648721270700128, 1]))

    # backward
    def exact(w: NDFloatArray) -> NDFloatArray:
        return np.exp(w**2) ** 2 * w * 4

    grad_w = Var(np.array([1.0]))
    grad_x = h.backward(grad_w)[0]
    grad_y = g.backward(grad_x)[0]
    grad_z = f.backward(grad_y)[0]
    assert np.allclose(grad_z.data, exact(input))


def test_backward_function() -> None:
    # forward
    f = Square()
    g = Exp()
    h = Square()

    input = np.array([0.5, 0])
    w = Var(input)
    x = f(w)
    y = g(x)
    z = h(y)
    assert np.allclose(z.data, np.array([1.648721270700128, 1]))

    # backward
    def exact(w: NDFloatArray) -> NDFloatArray:
        return np.exp(w**2) ** 2 * w * 4

    z.backward()
    assert np.allclose(w.grad.data.data, exact(input))


def test_functions() -> None:
    # forward
    x = Var(np.array([0.5, 0]))
    y = square(exp(square(x)))
    assert np.allclose(y.data, np.array([1.648721270700128, 1]))

    # backward
    def exact(w: NDFloatArray) -> NDFloatArray:
        return np.exp(w**2) ** 2 * w * 4

    y.backward()
    assert np.allclose(x.grad.data.data, exact(x.data))


def test_add() -> None:
    # forward
    x = Var(np.array(2))
    y = Var(np.array(3))
    z = add(x, y)
    assert z.data == 5.0

    # backward
    z.backward()
    assert x.grad.data == 1.0
    assert y.grad.data == 1.0

    # forward
    w = Var(np.array(1))
    x = Var(np.array(2))
    y = Var(np.array(3))
    z = add(add(w, x), y)
    assert z.data == 6.0


def test_add_same_variable() -> None:
    x = Var(np.array(3))
    y = add(x, x)
    y.backward()
    assert y.data == 6.0
    assert x.grad.data == 2.0


# @pytest.mark.xfail(reason="implement generation")
def test_add_same_variable_2() -> None:
    x = Var(np.array(3), name="x")
    a = add(x, x)
    a.name = "a"
    b = add(a, x)
    b.name = "b"
    y = add(b, x)
    y.name = "y"
    y.backward()
    assert y.data == 12.0
    assert x.grad.data == 4.0

    w = Var(np.array(3), "w")
    z = add(add(add(w, w), w), w)
    z.name = "z"
    z.backward()
    assert z.data == 12.0
    assert w.grad.data == 4.0


def test_square_and_add() -> None:
    x = Var(np.array(2.0))
    y = Var(np.array(3.0))
    z = add(square(x), square(y))
    z.backward()
    assert z.data == 13.0
    assert x.grad.data == 4.0
    assert y.grad.data == 6.0


def test_clear_grad() -> None:
    x = Var(np.array(3.0))
    y = add(x, x)
    y.backward()
    assert x.grad.data == 2.0

    x.clear_grad()
    y = add(add(x, x), x)
    y.backward()
    assert x.grad.data == 3.0


def test_mul() -> None:
    x = Var(np.array(3.0))
    y = Var(np.array(2.0))
    z = Var(np.array(1.0))
    w = add(mul(x, y), z)
    w.backward()
    assert w.data == 7.0
    assert x.grad.data == 2.0
    assert y.grad.data == 3.0


def test_generateion() -> None:
    x = Var(np.array(2.0), name="x")
    a = square(x)
    a.name = "a"
    b = square(a)
    b.name = "b"
    c = square(a)
    c.name = "c"
    y = add(b, c)
    y.name = "y"
    y.backward(retain_grad=True)
    assert y.data == 32.0
    assert b.grad.data == 1.0
    assert a.grad.data == 16.0
    assert x.grad.data == 64.0

    y.clear_grad()
    c.clear_grad()
    b.clear_grad()
    a.clear_grad()
    x.clear_grad()
    assert b.optional_grad is None
    assert a.optional_grad is None
    assert x.optional_grad is None

    y.backward(retain_grad=False)
    assert b.optional_grad is None
    assert a.optional_grad is None
    assert x.grad.data == 64.0


def test_diff() -> None:
    def f(x: Variable) -> Variable:
        return 4 * x * x * x + x

    x = Var(np.array(2.0))
    fx = diff_f(x, f, n=0)
    assert fx.data == 34

    dx = diff_f(x, f, n=1)
    assert dx.data == 49.0

    ddx = diff_f(x, f, n=2)
    assert ddx.data == 48.0

    dddx = diff_f(x, f, n=3)
    assert dddx.data == 24.0


def test_diff_exp() -> None:
    x = Var(np.array(2.0))
    fx = diff_f(x, exp, n=0)
    assert fx.data == np.exp(2)

    dx = diff_f(x, exp, n=1)
    assert dx.data == np.exp(2)

    ddx = diff_f(x, exp, n=2)
    assert ddx.data == np.exp(2)

    dddx = diff_f(x, exp, n=3)
    assert dddx.data == np.exp(2)


def test_diff_sin() -> None:
    x = Var(np.array(2.0))
    fx = diff_f(x, sin, n=0)
    assert fx.data == np.sin(2)

    dx = diff_f(x, sin, n=1)
    assert dx.data == np.cos(2)

    ddx = diff_f(x, sin, n=2)
    assert ddx.data == -np.sin(2)

    dddx = diff_f(x, sin, n=3)
    assert dddx.data == -np.cos(2)

    ddddx = diff_f(x, sin, n=4)
    assert ddddx.data == np.sin(2)


def test_diff_cos() -> None:
    x = Var(np.array(2.0))
    fx = diff_f(x, cos, n=0)
    assert fx.data == np.cos(2)

    dx = diff_f(x, cos, n=1)
    assert dx.data == -np.sin(2)

    ddx = diff_f(x, cos, n=2)
    assert ddx.data == -np.cos(2)

    dddx = diff_f(x, cos, n=3)
    assert dddx.data == np.sin(2)

    ddddx = diff_f(x, cos, n=4)
    assert ddddx.data == np.cos(2)


def test_diff_tanh() -> None:
    x = Var(np.array(2.0))
    fx = diff_f(x, tanh, n=0)
    assert fx.data == np.tanh(2)

    dx = diff_f(x, tanh, n=1)
    assert dx.data == 1 - np.tanh(2) * np.tanh(2)


def test_reshape() -> None:
    x = Var(np.random.rand(2, 3))
    y = reshape(x, (6,))
    y.backward(retain_grad=True)
    assert x.data.shape == (2, 3)
    assert np.all(x.grad.data == np.ones((2, 3)))


def test_transpose() -> None:
    origin = np.random.rand(2, 3)
    x = Var(origin.copy())
    y = transpose(x)
    y.backward(retain_grad=True)
    assert y.data.shape == (3, 2)
    assert x.data.shape == (2, 3)
    assert np.all(x.data == origin)
    assert np.all(x.grad.data == np.ones((2, 3)))


def test_sum() -> None:
    x = Var(np.array([1, 2, 3]))
    y = sum(x)
    y.backward(retain_grad=True)
    assert y.data.shape == (1,)
    assert x.data.shape == (3,)
    assert y.data == 6
    assert np.all(x.grad.data == np.ones(3))


def test_broadcast_to() -> None:
    origin = np.random.rand(3)
    x = Var(origin.copy())
    y = broadcast_to(x, (2, 3))
    y.backward(retain_grad=True)
    assert y.data.shape == (2, 3)
    assert x.data.shape == (3,)

    assert np.all(y.data[0] == origin)
    assert np.all(y.data[1] == origin)
    assert np.all(x.grad.data == np.ones(3) * 2)

    x.clear_grad()
    z = broadcast_to(x, (3,))
    z.backward(retain_grad=True)
    assert z.data.shape == (3,)
    assert x.data.shape == (3,)

    assert np.all(z.data == origin)
    assert np.all(x.grad.data == np.ones(3))


def test_sum_to() -> None:
    x = Var(np.array([[1, 2, 3], [4, 5, 6]]))
    y = sum_to(x, (3,))
    y.backward(retain_grad=True)
    assert y.data.shape == (3,)
    assert x.data.shape == (2, 3)

    assert np.all(y.data == np.array([5, 7, 9]))
    assert np.all(x.grad.data == np.ones((2, 3)))

    x.clear_grad()
    z = sum_to(x, (2, 3))
    z.backward(retain_grad=True)
    assert z.data.shape == (2, 3)
    assert x.data.shape == (2, 3)

    assert np.all(z.data == np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.all(x.grad.data == np.ones((2, 3)))


def test_matmul() -> None:
    x = Var(np.array([1, 2]))
    w = Var(np.array([2, 3]))
    y = matmul(x, w)
    y.backward()
    assert y.data == 8
    assert np.all(x.grad.data == np.array([2, 3]))
    assert np.all(w.grad.data == np.array([1, 2]))
