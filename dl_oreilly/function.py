from abc import ABC, abstractmethod
from typing import Callable, cast

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
    def x(self) -> Variable:
        return self._input

    @property
    def output(self) -> Variable:
        return self._output

    @property
    def inputs(self) -> tuple[Variable, ...]:
        return (self._input,)

    @abstractmethod
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _backward_core(self, x: Variable) -> Variable:
        ...

    def backward(self, grad: Variable) -> tuple[Variable, ...]:
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
    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        ...

    def backward(self, grad: Variable) -> tuple[Variable, ...]:
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

    def _backward_core(self, grad: Variable) -> Variable:
        return (2 * self.x) * grad


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

    def _backward_core(self, grad: Variable) -> Variable:
        return exp(self.x) * grad


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

    def _backward_core(self, grad: Variable) -> Variable:
        return -grad


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

    def _backward_core(self, grad: Variable) -> Variable:
        return self._exp * (self.x ** (self._exp - 1)) * grad


class Sin(OneArgFunction):
    """
    f(x) = sin(x)
    f'(x) = cos(x)
    """

    @property
    def name(self) -> str:
        return "sin"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.sin(x)

    def _backward_core(self, grad: Variable) -> Variable:
        return cos(self.x) * grad


class Cos(OneArgFunction):
    """
    f(x) = cos(x)
    f'(x) = -sin(x)
    """

    @property
    def name(self) -> str:
        return "cos"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.cos(x)

    def _backward_core(self, grad: Variable) -> Variable:
        return -sin(self.x) * grad


class Tanh(OneArgFunction):
    """
    f(x) = tanh(x)
    f'(x) = 1 -y*2
    """

    @property
    def name(self) -> str:
        return "tanh"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.tanh(x)

    def _backward_core(self, grad: Variable) -> Variable:
        y = self.output
        return grad * (1 - y * y)


class Reshape(OneArgFunction):
    """
    Example.
    foward   : np.array([[[1,2],[3,4]]) -> [1, 2, 3, 4]
    backward : [1, 2, 3, 4] -> np.array([[[1,2],[3,4]])
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    @property
    def name(self) -> str:
        return "reshape"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        self._xshape = x.shape
        return x.reshape(self._shape)

    def _backward_core(self, grad: Variable) -> Variable:
        return reshape(grad, self._xshape)


class Transpose(OneArgFunction):
    """
    Example.
    foward   : np.array([[[1,2],[3,4]]) -> np.array([[1,3],[2,4]])
    backward : np.array([[[1,3],[2,4]]) -> np.array([[1,2],[3,4]])
    """

    @property
    def name(self) -> str:
        return "transpose"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.transpose(x)

    def _backward_core(self, grad: Variable) -> Variable:
        return transpose(grad)


class Sum(OneArgFunction):
    """
    Example.
    foward   : np.array([[[1,2],[3,4]]) -> 10
    backward : 10 -> np.array([[1,2],[3,4]])
    """

    @property
    def name(self) -> str:
        return "sum"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        self._xshape = x.shape
        return np.array([np.sum(x)])

    def _backward_core(self, grad: Variable) -> Variable:
        return broadcast_to(grad, self._xshape)


class BroadcastTo(OneArgFunction):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    @property
    def name(self) -> str:
        return "broadcastTo"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        self._xshape = x.shape
        return np.broadcast_to(x, self._shape)

    def _backward_core(self, grad: Variable) -> Variable:
        return sum_to(grad, self._xshape)


class SumTo(OneArgFunction):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    @property
    def name(self) -> str:
        return "sumTo"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        self._xshape = x.shape
        ndim = len(self._shape)
        lead = x.ndim - ndim
        assert lead >= 0, "invalid argument"
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(self._shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        else:
            y = np.array(y)
        return cast(NDFloatArray, y)

    def _backward_core(self, grad: Variable) -> Variable:
        return broadcast_to(grad, self._xshape)


class Add(TwoArgsFunction):
    """
    f(x, y) = x + y
    """

    @property
    def name(self) -> str:
        return "add"

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        return x + y

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
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

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
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

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        x1, x2 = self.inputs
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

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        x1, x2 = self.inputs
        return (grad / x2, -grad * x1 / x2 / x2)


class MatMul(TwoArgsFunction):
    """
    f(x, y) = x @ y
    """

    @property
    def name(self) -> str:
        return "matmul"

    def forward(self, x: NDFloatArray, W: NDFloatArray) -> NDFloatArray:
        ret = x.dot(W)
        if ret.ndim == 0:
            return np.array(ret)
        return cast(NDFloatArray, ret)

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        x, W = self.inputs
        assert (
            x.ndim == W.ndim
        ), f"x.ndim({x.ndim}) must be same as W.ndim({W.ndim}). Otherwise, transpose not working expctedly."
        gx = matmul(grad, W.T)
        gw = matmul(x.T, grad)
        return (gx, gw)


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


def sin(x: Variable) -> Variable:
    return Sin()(x)


def cos(x: Variable) -> Variable:
    return Cos()(x)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


def reshape(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return x
    return Reshape(shape)(x)


def transpose(x: Variable) -> Variable:
    return Transpose()(x)


def sum(x: Variable) -> Variable:
    return Sum()(x)


def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    return BroadcastTo(shape)(x)


def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    return SumTo(shape)(x)


def matmul(x: Variable, W: Variable) -> Variable:
    return MatMul()(x, W)


def diff_f(x: Variable, f: Callable[[Variable], Variable], n: int = 1) -> Variable:
    create_graph = True
    y = f(x)

    while n > 0:
        x.clear_grad()
        if n == 1:
            create_graph = False
        y.backward(create_graph=create_graph)
        n -= 1

        y = x.grad

    return y
