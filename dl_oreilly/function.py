from abc import ABC, abstractmethod

import numpy as np

from . import NDFloatArray
from .protocol import Variable


class OneArgFunction(ABC):
    def __call__(self, input: Variable) -> Variable:
        self._input = input
        assert getattr(self, "_generation", None) is None, "this function has already been called. but called again!"
        self._generation = input.generation
        y = self.forward(input.data)

        output = input.new_variable(y)
        output.creator = self

        self._output = output
        return output

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def input(self) -> Variable:
        return self._input

    @property
    def output(self) -> Variable:
        return self._output

    @property
    def inputs(self) -> tuple[Variable, ...]:
        return (self._input,)

    @property
    def x(self) -> NDFloatArray:
        return self._input.data

    @abstractmethod
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        ...

    @abstractmethod
    def _backward_core(self, x: NDFloatArray) -> NDFloatArray:
        ...

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        return (self._backward_core(grad),)


class TwoArgsFunction(ABC):
    def __call__(self, x1: Variable, x2: Variable) -> Variable:
        self._inputs = (x1, x2)
        assert getattr(self, "_generation", None) is None, "this function has already been called. but called again!"
        self._generation = max(x1.generation, x2.generation)
        y = self.forward(x1.data, x2.data)

        output = x1.new_variable(y)
        output.creator = self

        self._output = output
        return output

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def inputs(self) -> tuple[Variable, Variable]:
        return self._inputs

    @property
    def xs(self) -> tuple[NDFloatArray, NDFloatArray]:
        return (self._inputs[0].data, self._inputs[1].data)

    @property
    def output(self) -> Variable:
        return self._output

    @abstractmethod
    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        ...

    @abstractmethod
    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        ...

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        return self._backward_core(grad)


class Square(OneArgFunction):
    """
    f(x) = x^2
    f'(x) = 2x
    """

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return x**2

    def _backward_core(self, grad_y: NDFloatArray) -> NDFloatArray:
        grad_x = (2 * self.x) * grad_y
        return grad_x


class Exp(OneArgFunction):
    """
    f(x) = exp(x)
    f'(x) = exp(x)
    """

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.exp(x)

    def _backward_core(self, grad_y: NDFloatArray) -> NDFloatArray:
        grad_x = np.exp(self.x) * grad_y
        return grad_x


class Neg(OneArgFunction):
    """
    f(x) = -x
    f'(x) = -1
    """

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return -x

    def _backward_core(self, grad_y: NDFloatArray) -> NDFloatArray:
        return -grad_y


class Add(TwoArgsFunction):
    """
    f(x, y) = x + y
    """

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x + y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        return (grad, grad)


class Sub(TwoArgsFunction):
    """
    f(x, y) = x - y
    """

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x - y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        return (grad, -grad)


class Mul(TwoArgsFunction):
    """
    f(x, y) = x * y
    """

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x * y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        x1, x2 = self.xs
        return (grad * x2, grad * x1)


class Div(TwoArgsFunction):
    """
    f(x, y) = x / y
    """

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x / y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        x1, x2 = self.xs
        return (grad / x2, -grad * x1 / x2 / x2)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def neg(x: Variable) -> Variable:
    return Neg()(x)


def add(x1: Variable, x2: Variable) -> Variable:
    return Add()(x1, x2)


def sub(x1: Variable, x2: Variable) -> Variable:
    return Sub()(x1, x2)


def mul(x1: Variable, x2: Variable) -> Variable:
    return Mul()(x1, x2)


def div(x1: Variable, x2: Variable) -> Variable:
    return Div()(x1, x2)
