from functools import reduce

import numpy as np

from . import NDFloatArray
from .variable import Function, Variable, VariadicArgsFunction


class Square(Function):
    """
    f(x) = x^2
    f'(x) = 2x
    """

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return x**2

    def backward(self, grad_y: NDFloatArray) -> NDFloatArray:
        grad_x = (2 * self.x) * grad_y
        return grad_x


class Exp(Function):
    """
    f(x) = exp(x)
    f'(x) = exp(x)
    """

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.exp(x)

    def backward(self, grad_y: NDFloatArray) -> NDFloatArray:
        grad_x = np.exp(self.x) * grad_y
        return grad_x


class Add(VariadicArgsFunction):
    """
    f(x, y, z, ...) = x + y + z + ...
    """

    def forward(self, xs: tuple[NDFloatArray, ...]) -> NDFloatArray:
        return reduce(lambda x, y: x + y, xs, np.array(0.0))

    def backward(self, grad_y: NDFloatArray) -> tuple[NDFloatArray, ...]:
        return tuple([grad_y] * len(self.inputs))


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def add(*xs: Variable) -> Variable:
    return Add()(*xs)
