import matplotlib.pyplot as plt
import numpy as np

from . import NDFloatArray
from .function import mean_squared_error, sigmoid
from .layers import Linear
from .protocol import Variable
from .variable import Var


def _save_figure(x: NDFloatArray, y: NDFloatArray, save: bool) -> None:
    if not save:
        return

    plt.figure(figsize=(10, 7.5))
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig("plot.png")


def test_simple_nn(save: bool = True) -> None:
    np.random.seed(0)
    x = Var(np.random.rand(100, 1), name="x")
    y = Var(np.sin(2 * np.pi * x.data) + np.random.rand(100, 1), name="y")

    l1 = Linear(10)
    l2 = Linear(1)

    def predict(x: Variable) -> Variable:
        return l2(sigmoid(l1(x)))

    lr = 0.2
    iters = 1000

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        l1.clear_grad()
        l2.clear_grad()
        loss.backward()

        for ll in [l1, l2]:
            for p in ll.params():
                print(p.name)
                print("shape", p.grad.data.shape, p.data.shape)
                p.data -= lr * p.grad.data

    _save_figure(x.data, y.data, save)
