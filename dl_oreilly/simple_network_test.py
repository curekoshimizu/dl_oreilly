import matplotlib.pyplot as plt
import numpy as np
import pytest

from . import NDFloatArray
from .function import mean_squared_error, sigmoid
from .layers import Linear
from .models import MLP, TwoLayerNet
from .optimizers import SGD
from .protocol import Variable
from .variable import Var


def _save_figure(x: NDFloatArray, y_pred: NDFloatArray, y: NDFloatArray, save: bool) -> None:
    if not save:
        return

    plt.figure(figsize=(10, 7.5))
    plt.scatter(x, y_pred, c="green")
    plt.scatter(x, y, c="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig("plot.png")


@pytest.mark.heavy
def test_simple_nn(save: bool = False) -> None:
    np.random.seed(0)
    x = Var(np.random.rand(100, 1), name="x")
    y = Var(np.sin(2 * np.pi * x.data) + np.random.rand(100, 1), name="y")

    l1 = Linear(10)
    l2 = Linear(1)

    def predict(x: Variable) -> Variable:
        return l2(sigmoid(l1(x)))

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        l1.clear_grad()
        l2.clear_grad()
        loss.backward()

        for ll in [l1, l2]:
            for p in ll.params():
                p.data -= lr * p.grad.data

    assert 0.01 < loss.data < 0.1

    _save_figure(x.data, y_pred.data, y.data, save)


def test_simple_nn_check_count(save: bool = False) -> None:
    np.random.seed(0)
    x = Var(np.random.rand(100, 1), name="x")
    y = Var(np.sin(2 * np.pi * x.data) + np.random.rand(100, 1), name="y")

    l1 = Linear(10)
    l2 = Linear(1)

    def predict(x: Variable) -> Variable:
        return l2(sigmoid(l1(x)))

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        l1.clear_grad()
        l2.clear_grad()
        loss.backward()
        if loss.data < 0.25:
            break

        for ll in [l1, l2]:
            for p in ll.params():
                p.data -= lr * p.grad.data
    assert i == 968


def test_two_layer_check_count(save: bool = False) -> None:
    np.random.seed(0)
    x = Var(np.random.rand(100, 1), name="x")
    y = Var(np.sin(2 * np.pi * x.data) + np.random.rand(100, 1), name="y")

    model = TwoLayerNet(hidden_size=10, out_size=1)

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        model.clear_grad()
        loss.backward()
        if loss.data < 0.25:
            break

        for p in model.params():
            p.data -= lr * p.grad.data
    assert i == 968


def test_mlp_check_count(save: bool = False) -> None:
    np.random.seed(0)
    x = Var(np.random.rand(100, 1), name="x")
    y = Var(np.sin(2 * np.pi * x.data) + np.random.rand(100, 1), name="y")

    model = MLP(fc_output_sizes=(10, 1))
    optimizer = SGD(lr=0.2).setup(model)

    iters = 10000

    for i in range(iters):
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        model.clear_grad()
        loss.backward()
        if loss.data < 0.25:
            break
        optimizer.update()
    assert i == 968
