from typing import Callable

import numpy as np
import pytest

from .protocol import Variable
from .variable import Var


def _rosenblock_loop(update: Callable[[Variable, Variable], None], create_graph: bool) -> int:
    def rosenblock(x0: Variable, v1: Variable) -> Variable:
        return 100 * (x1 - x0 * x0) ** 2 + (x0 - 1) * (x0 - 1)

    x0: Variable = Var(np.array(0.0), name="x0")
    x1: Variable = Var(np.array(2.0), name="x1")

    max_loop = 100000

    for i in range(max_loop):
        y = rosenblock(x0, x1)
        x0.clear_grad()
        x1.clear_grad()
        y.backward(create_graph=create_graph)

        if np.allclose([x0.data, x1.data], [1.0, 1.0]):
            break

        update(x0, x1)
    return i


@pytest.mark.heavy
def test_normal_grad_with_rosenblock() -> None:
    lr = 0.001

    def update(x0: Variable, x1: Variable) -> None:
        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data

    n = _rosenblock_loop(update, create_graph=False)
    assert n == 27514


def _simple_func_loop(update: Callable[[Variable], None], create_graph: bool) -> int:
    def simple_func(x: Variable) -> Variable:
        return x**4 - 2 * x * x

    x: Variable = Var(np.array(2.0), name="x")

    max_loop = 100000

    for i in range(max_loop):
        y = simple_func(x)
        x.clear_grad()
        y.backward(create_graph=create_graph)

        if np.allclose(x.data, [1.0]):
            break

        update(x)
    return i


def test_normal_grad() -> None:
    lr = 0.01

    def update(x: Variable) -> None:
        x.data -= lr * x.grad.data

    n = _simple_func_loop(update, create_graph=False)
    assert n == 126


def test_newton_grad() -> None:
    lr = 0.001

    def update(x: Variable) -> None:
        x.data -= lr * x.grad.data
        gx = x.grad
        x.clear_grad()
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data

    n = _simple_func_loop(update, create_graph=True)
    assert n == 5
