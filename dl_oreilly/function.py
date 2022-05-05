import numpy as np

from . import NDFloatArray
from .variable import Function, Variable


class Square(Function):
    """
    f(x) = x^2
    f'(x) = 2x
    """

    def forward(self, xs: tuple[NDFloatArray, ...]) -> tuple[NDFloatArray, ...]:
        return tuple(self._forward(x) for x in xs)

    def _forward(self, x: NDFloatArray) -> NDFloatArray:
        return x**2

    def backward(self, grad_y: NDFloatArray) -> NDFloatArray:
        grad_x = (2 * self.x) * grad_y
        return grad_x


class Exp(Function):
    """
    f(x) = exp(x)
    f'(x) = exp(x)
    """

    def forward(self, xs: tuple[NDFloatArray, ...]) -> tuple[NDFloatArray, ...]:
        return tuple(self._forward(x) for x in xs)

    def _forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.exp(x)

    def backward(self, grad_y: NDFloatArray) -> NDFloatArray:
        grad_x = np.exp(self.x) * grad_y
        return grad_x


class Add(Function):
    """
    f(x, y) = x + y
    """

    def forward(self, x: tuple[NDFloatArray, ...]) -> tuple[NDFloatArray, ...]:
        x0, x1 = x
        return (x0 + x1,)

    def backward(self, grad_y: NDFloatArray) -> NDFloatArray:
        raise NotImplementedError()


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def add(x: Variable, y: Variable) -> Variable:
    return Add()(x, y)
