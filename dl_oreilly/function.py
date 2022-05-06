import numpy as np

from . import NDFloatArray
from .variable import Function, TwoArgsFunction, Variable


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


class Add(TwoArgsFunction):
    """
    f(x, y) = x + y
    """

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x + y

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        return (grad, grad)


class Mul(TwoArgsFunction):
    """
    f(x, y) = x * y
    """

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x * y

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        x1, x2 = self.xs
        return (grad * x2, grad * x1)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def add(x1: Variable, x2: Variable) -> Variable:
    return Add()(x1, x2)


def mul(x1: Variable, x2: Variable) -> Variable:
    return Mul()(x1, x2)
