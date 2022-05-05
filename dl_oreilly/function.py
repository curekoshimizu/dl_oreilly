from abc import ABC, abstractmethod

import numpy as np

from . import NDFloatArray
from .variable import Variable


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        self._input = input
        x = input.data
        y = self.forward(x)
        return Variable(y)

    @property
    def x(self) -> NDFloatArray:
        return self._input.data

    @abstractmethod
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        ...

    @abstractmethod
    def backward(self, x: NDFloatArray) -> NDFloatArray:
        ...


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
