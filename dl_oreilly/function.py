from abc import ABC, abstractmethod

import numpy as np

from . import NDFloatArray
from .config import Config
from .protocol import Variable


class OneArgFunction(ABC):
    def __call__(self, input: Variable) -> Variable:
        self._input = input
        assert getattr(self, "_generation", None) is None, "this function has already been called. but called again!"
        y = self.forward(input.data)

        output = input.new_variable(y)

        if Config.enable_backprop:
            self._generation = input.generation
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

    @property
    @abstractmethod
    def name(self) -> str:
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
        y = self.forward(x1.data, x2.data)

        output = x1.new_variable(y)

        if Config.enable_backprop:
            self._generation = max(x1.generation, x2.generation)
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

    @property
    @abstractmethod
    def name(self) -> str:
        ...

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

    @property
    def name(self) -> str:
        return "square"

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

    @property
    def name(self) -> str:
        return "exp"

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

    @property
    def name(self) -> str:
        return "neg"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return -x

    def _backward_core(self, grad_y: NDFloatArray) -> NDFloatArray:
        return -grad_y


class Pow(OneArgFunction):
    """
    f(x) = x ** exp
    f'(x) = exp * x ** (exp - 1)
    """

    def __init__(self, exp: Variable) -> None:
        self._exp = exp.data

    @property
    def name(self) -> str:
        return "pow"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return x**self._exp

    def _backward_core(self, grad_y: NDFloatArray) -> NDFloatArray:
        return self._exp * (self.x ** (self._exp - 1)) * grad_y


class Add(TwoArgsFunction):
    """
    f(x, y) = x + y
    """

    @property
    def name(self) -> str:
        return "add"

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x + y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        return (grad, grad)


class Sub(TwoArgsFunction):
    """
    f(x, y) = x - y
    """

    @property
    def name(self) -> str:
        return "sub"

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x - y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        return (grad, -grad)


class Mul(TwoArgsFunction):
    """
    f(x, y) = x * y
    """

    @property
    def name(self) -> str:
        return "mul"

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x * y

    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        x1, x2 = self.xs
        return (grad * x2, grad * x1)


class Div(TwoArgsFunction):
    """
    f(x, y) = x / y
    """

    @property
    def name(self) -> str:
        return "div"

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


def pow(base: Variable, exp: Variable) -> Variable:
    return Pow(exp)(base)
