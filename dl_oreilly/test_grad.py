from typing import Callable

import numpy as np
import pytest

from .protocol import Variable
from .variable import Var


def _logic(update: Callable[[Variable, Variable], None]) -> int:
    def rosenblock(x0: Variable, v1: Variable) -> Variable:
        return 100 * (x1 - x0 * x0) ** 2 + (x0 - 1) * (x0 - 1)

    x0: Variable = Var(np.array(0.0), name="x0")
    x1: Variable = Var(np.array(2.0), name="x1")

    max_loop = 100000

    for i in range(max_loop):
        y = rosenblock(x0, x1)
        x0.clear_grad()
        x1.clear_grad()
        y.backward()

        if np.allclose([x0.data, x1.data], [1.0, 1.0]):
            break

        update(x0, x1)
    return i


@pytest.mark.heavy
def test_normal_grad() -> None:
    lr = 0.001

    def update(x0: Variable, x1: Variable) -> None:
        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

    n = _logic(update)
    assert n == 27514
