import numpy as np

from . import NDFloatArray
from .config import use_test_mode
from .function import (
    Exp,
    Square,
    accuracy,
    add,
    broadcast_to,
    cos,
    diff_f,
    dropout,
    exp,
    get_conv_outsize,
    get_item,
    matmul,
    mean_squared_error,
    mul,
    reshape,
    sin,
    softmax,
    softmax1d,
    softmax_cross_entropy,
    square,
    sum,
    sum_to,
    tanh,
    transpose,
)
from .models import MLP
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

    grad_w = Var(np.array([1.0, 1.0]))
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
    x = Var(np.array([2]))
    y = Var(np.array([3]))
    z = add(x, y)
    assert z.data == 5.0

    # backward
    z.backward()
    assert x.grad.data == 1.0
    assert y.grad.data == 1.0

    # forward
    w = Var(np.array([1]))
    x = Var(np.array([2]))
    y = Var(np.array([3]))
    z = add(add(w, x), y)
    assert z.data == 6.0


def test_add_same_variable() -> None:
    x = Var(np.array([3]))
    y = add(x, x)
    y.backward()
    assert y.data == 6.0
    assert x.grad.data == 2.0


# @pytest.mark.xfail(reason="implement generation")
def test_add_same_variable_2() -> None:
    x = Var(np.array([3]), name="x")
    a = add(x, x)
    a.name = "a"
    b = add(a, x)
    b.name = "b"
    y = add(b, x)
    y.name = "y"
    y.backward()
    assert y.data == 12.0
    assert x.grad.data == 4.0

    w = Var(np.array([3]), "w")
    z = add(add(add(w, w), w), w)
    z.name = "z"
    z.backward()
    assert z.data == 12.0
    assert w.grad.data == 4.0


def test_square_and_add() -> None:
    x = Var(np.array([2.0]))
    y = Var(np.array([3.0]))
    z = add(square(x), square(y))
    z.backward()
    assert z.data == 13.0
    assert x.grad.data == 4.0
    assert y.grad.data == 6.0


def test_clear_grad() -> None:
    x = Var(np.array([3.0]))
    y = add(x, x)
    y.backward()
    assert x.grad.data == 2.0

    x.clear_grad()
    y = add(add(x, x), x)
    y.backward()
    assert x.grad.data == 3.0


def test_mul() -> None:
    x = Var(np.array([3.0]))
    y = Var(np.array([2.0]))
    z = Var(np.array([1.0]))
    w = add(mul(x, y), z)
    w.backward()
    assert w.data == 7.0
    assert x.grad.data == 2.0
    assert y.grad.data == 3.0


def test_generateion() -> None:
    x = Var(np.array([2.0]), name="x")
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

    x = Var(np.array([2.0]))
    fx = diff_f(x, f, n=0)
    assert fx.data == 34

    dx = diff_f(x, f, n=1)
    assert dx.data == 49.0

    ddx = diff_f(x, f, n=2)
    assert ddx.data == 48.0

    dddx = diff_f(x, f, n=3)
    assert dddx.data == 24.0


def test_diff_exp() -> None:
    x = Var(np.array([2.0]))
    fx = diff_f(x, exp, n=0)
    assert fx.data == np.exp(2)

    dx = diff_f(x, exp, n=1)
    assert dx.data == np.exp(2)

    ddx = diff_f(x, exp, n=2)
    assert ddx.data == np.exp(2)

    dddx = diff_f(x, exp, n=3)
    assert dddx.data == np.exp(2)


def test_diff_sin() -> None:
    x = Var(np.array([2.0]))
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
    x = Var(np.array([2.0]))
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
    x = Var(np.array([2.0]))
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

    x = Var(np.array([1, 2, 3]))
    y = transpose(x)
    y.backward()
    assert y.data.shape == (3,)
    assert x.data.shape == (3,)
    assert np.all(x.data == y.data)
    assert np.all(x.grad.data == np.ones((2, 3)))


def test_sum() -> None:
    x = Var(np.array([1, 2, 3]))
    y = sum(x)
    y.backward(retain_grad=True)
    assert y.data.shape == (1,)
    assert x.data.shape == (3,)
    assert y.data == 6
    assert np.all(x.grad.data == np.ones(3))

    x = Var(np.array([[1, 2, 3], [4, 5, 6]]))
    y = sum(x, keepdims=True)
    assert y.data == 21
    assert y.shape == (1, 1)

    x = Var(np.array([[1, 2, 3], [4, 5, 6]]))
    y = sum(x, axis=0)
    assert np.allclose(y.data, np.array([5, 7, 9]))
    y.backward()
    assert np.allclose(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]]))

    x = Var(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    assert y.shape == (1, 1, 1, 1)


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


def test_get_item() -> None:
    x = Var(np.array([[1, 2, 3], [4, 5, 6]]))
    y = get_item(x, 1)
    assert np.all(y.data == np.array([4, 5, 6]))
    y.backward()
    assert np.all(x.grad.data == np.array([[0, 0, 0], [1, 1, 1]]))


def test_matmul_ndim1() -> None:
    x = Var(np.array([1, 2]))
    w = Var(np.array([2, 3]))
    y = matmul(x, w)
    y.backward()
    assert y.data == 8
    assert np.all(x.grad.data == np.array([2, 3]))
    assert np.all(w.grad.data == np.array([1, 2]))


def test_matmul_ndim2() -> None:
    x = Var(np.array([[1, 2]]), name="x")
    w = Var(np.array([[2, 3], [4, 5]]), name="w")
    y = matmul(x, w)
    y.name = "y"
    z = y.sum()
    z.name = "z"
    z.backward()
    assert np.all(y.data == np.array([10, 13]))
    assert np.all(z.data == np.array([23]))
    assert z.data == 23
    assert np.all(x.grad.data == np.array([5, 9]))
    assert np.all(w.grad.data == np.array([[1, 1], [2, 2]]))


def test_softmax1d() -> None:
    x = Var(np.array([-0.61505778, -0.4290161, 0.31733289]))
    y = softmax1d(x)
    assert np.sum(y.data) == 1
    assert np.allclose(y.data, np.array([0.21074602, 0.25383778, 0.5354162]))


def test_softmax() -> None:
    x = Var(
        np.array(
            [
                [-0.615, -0.427, 0.317],
                [-0.763, -0.249, 0.185],
                [-0.520, -0.962, 0.578],
                [-0.942, -0.503, 0.175],
            ]
        )
    )
    y = softmax(x)
    for i in range(4):
        assert np.isclose(np.sum(y.data[i]), 1)


def test_mean_squared_error() -> None:
    x1 = Var(np.array([1.0, 2.0, 3.0]))
    y1 = Var(np.array([1.1, 2.2, 3.3]))

    z1 = mean_squared_error(x1, y1)
    z1.backward()

    x2 = Var(np.array([1.0, 2.0, 3.0]))
    y2 = Var(np.array([1.1, 2.2, 3.3]))

    diff = x2 - y2
    z2 = (diff * diff).sum() / 3
    z2.backward()

    assert np.allclose(x1.data, x2.data)
    assert np.allclose(y1.data, y2.data)
    assert np.allclose(z1.data, z2.data)
    assert np.allclose(x1.grad.data, x2.grad.data)
    assert np.allclose(y1.grad.data, y2.grad.data)


def test_soft_cross_entropy() -> None:
    np.random.seed(0)
    model = MLP((10, 3))

    x = Var(np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
    t = Var(np.array([2, 0, 1, 0]))
    y = model(x)
    loss = softmax_cross_entropy(y, t)
    assert np.isclose(loss.data, 1.4967442524053058)


def test_accuracy() -> None:
    y = Var(
        np.array(
            [
                [0.2, 0.8, 0],
                [0.1, 0.9, 0],
                [0.8, 0.1, 0.1],
            ]
        )
    )
    t = Var(np.array([1, 2, 0]))
    acc = accuracy(y, t)
    assert np.isclose(acc.data, 2.0 / 3.0)


def test_dropout() -> None:
    n = 100
    np.random.seed(0)
    x = Var(np.ones(n))
    y = dropout(x)
    assert abs(y.sum().data[0] - n) < 5  # 5% error
    with use_test_mode():
        z = dropout(x)
        assert np.all(z.data == x.data)


def test_get_conv_outsize() -> None:
    assert get_conv_outsize(input_size=4, kernel_size=3, stride=1, pad=0) == 2
    assert get_conv_outsize(input_size=4, kernel_size=3, stride=1, pad=1) == 4
    assert get_conv_outsize(input_size=7, kernel_size=3, stride=2, pad=0) == 3
