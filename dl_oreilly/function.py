from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, cast

import numpy as np
from numpy.typing import NDArray

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


class Log(OneArgFunction):
    """
    f(x) = log(x)
    f'(x) = 1/x
    """

    @property
    def name(self) -> str:
        return "log"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.log(x)

    def _backward_core(self, grad: Variable) -> Variable:
        return grad / self.x


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
        self._exp = exp

    @property
    def name(self) -> str:
        return "pow"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return x**self._exp.data

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

    def __init__(self, axis: Optional[int], keepdims: bool) -> None:
        self._axis = axis
        self._keepdims = keepdims

    @property
    def name(self) -> str:
        return "sum"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        self._xshape = x.shape
        y = x.sum(axis=self._axis, keepdims=self._keepdims)
        if np.isscalar(y):
            return np.array([y])
        return cast(NDFloatArray, y)

    def _backward_core(self, grad: Variable) -> Variable:
        grad = _reshape_sum_backward(grad, self._xshape, self._axis, self._keepdims)
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


class Clip(OneArgFunction):
    def __init__(self, x_min: float, x_max: float) -> None:
        self._x_min = x_min
        self._x_max = x_max

    @property
    def name(self) -> str:
        return "clip"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        y = np.clip(x, self._x_min, self._x_max)
        return cast(NDFloatArray, y)

    def _backward_core(self, gy: Variable) -> Variable:
        x = self._input
        mask = (x.data >= self._x_min) * (x.data <= self._x_max)
        gx = gy * cast(NDFloatArray, mask)
        return gx


class GetItem(OneArgFunction):
    def __init__(self, slices: int | tuple[NDArray[Any], ...]) -> None:
        self._slices = slices

    @property
    def name(self) -> str:
        return "get_item"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        y = x[self._slices]
        if np.isscalar(y):
            return np.array(y)
        return cast(NDFloatArray, y)

    def _backward_core(self, grad: Variable) -> Variable:
        x = self.x
        f = GetItemGrad(self._slices, x.shape)
        return f(grad)


class GetItemGrad(OneArgFunction):
    def __init__(self, slices: int | tuple[NDArray[Any], ...], in_shape: tuple[int, ...]) -> None:
        self._slices = slices
        self._in_shape = in_shape

    @property
    def name(self) -> str:
        return "get_item_grad"

    def forward(self, gy: NDFloatArray) -> NDFloatArray:
        gx = np.zeros(self._in_shape)
        np.add.at(gx, self._slices, gy)
        return gx

    def _backward_core(self, ggx: Variable) -> Variable:
        return get_item(ggx, self._slices)


class ReLU(OneArgFunction):
    @property
    def name(self) -> str:
        return "relu"

    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.maximum(x, 0.0)

    def _backward_core(self, grad: Variable) -> Variable:
        x = self.x
        mask = x.data > 0
        return grad * mask  # type:ignore


class MeanSquaredError(TwoArgsFunction):
    @property
    def name(self) -> str:
        return "mean_squared_error"

    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        diff = x - y
        ret = (diff * diff).sum() / len(diff)
        if np.isscalar(ret):
            return np.array([ret])
        return cast(NDFloatArray, ret)

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(grad, diff.shape)
        gx0 = gy * diff * (2 / len(diff))
        gx1 = -gx0
        assert gx0.shape == x0.shape
        assert gx1.shape == x1.shape
        return gx0, gx1


class Add(TwoArgsFunction):
    """
    f(x, y) = x + y
    """

    @property
    def name(self) -> str:
        return "add"

    def forward(self, x0: NDFloatArray, x1: NDFloatArray) -> NDFloatArray:
        self._x0shape = x0.shape
        self._x1shape = x1.shape
        return x0 + x1

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        gx0, gx1 = grad, grad
        if self._x0shape != self._x1shape:  # for broadcaset
            gx0 = sum_to(gx0, self._x0shape)
            gx1 = sum_to(gx1, self._x1shape)
        return gx0, gx1


class Sub(TwoArgsFunction):
    """
    f(x, y) = x - y
    """

    @property
    def name(self) -> str:
        return "sub"

    def forward(self, x0: NDFloatArray, x1: NDFloatArray) -> NDFloatArray:
        self._x0shape = x0.shape
        self._x1shape = x1.shape
        return x0 - x1

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        gx0, gx1 = grad, -grad
        if self._x0shape != self._x1shape:  # for broadcaset
            gx0 = sum_to(gx0, self._x0shape)
            gx1 = sum_to(gx1, self._x1shape)
        return gx0, gx1


class Mul(TwoArgsFunction):
    """
    f(x, y) = x * y
    """

    @property
    def name(self) -> str:
        return "mul"

    def forward(self, x0: NDFloatArray, x1: NDFloatArray) -> NDFloatArray:
        self._x0shape = x0.shape
        self._x1shape = x1.shape
        return x0 * x1

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0, gx1 = grad * x1, grad * x0
        if self._x0shape != self._x1shape:  # for broadcaset
            gx0 = sum_to(gx0, self._x0shape)
            gx1 = sum_to(gx1, self._x1shape)
        return gx0, gx1


class Div(TwoArgsFunction):
    """
    f(x, y) = x / y
    """

    @property
    def name(self) -> str:
        return "div"

    def forward(self, x0: NDFloatArray, x1: NDFloatArray) -> NDFloatArray:
        self._x0shape = x0.shape
        self._x1shape = x1.shape
        return x0 / x1

    def _backward_core(self, grad: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0, gx1 = grad / x1, grad * (-x0 / x1 / x1)
        if self._x0shape != self._x1shape:  # for broadcaset
            gx0 = sum_to(gx0, self._x0shape)
            gx1 = sum_to(gx1, self._x1shape)
        return gx0, gx1


class MatMul(TwoArgsFunction):
    """
    f(x, y) = x @ y
    """

    @property
    def name(self) -> str:
        return "matmul"

    def forward(self, x: NDFloatArray, W: NDFloatArray) -> NDFloatArray:
        ret = x.dot(W)
        if np.isscalar(ret):
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


def _reshape_sum_backward(grad: Variable, xshape: tuple[int, ...], axis: Optional[int], keepdims: bool) -> Variable:
    ndim = len(xshape)

    shape: tuple[int, ...]
    if not (ndim == 0 or axis is None or keepdims):
        if axis < 0:
            raise NotImplementedError()

        shape_list = list(grad.shape)
        shape_list.insert(axis, 1)
        shape = tuple(shape_list)
    else:
        shape = grad.shape
    return grad.reshape(shape)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def log(x: Variable) -> Variable:
    return Log()(x)


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


def sum(x: Variable, axis: Optional[int] = None, keepdims: bool = False) -> Variable:
    return Sum(axis, keepdims)(x)


def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    return BroadcastTo(shape)(x)


def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    return SumTo(shape)(x)


def clip(x: Variable, x_min: float, x_max: float) -> Variable:
    return Clip(x_min, x_max)(x)


def get_item(x: Variable, slices: int | tuple[NDArray[Any], ...]) -> Variable:
    return GetItem(slices)(x)


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


def linear(x: Variable, W: Variable, b: Optional[Variable]) -> Variable:
    t = matmul(x, W)
    if b is None:
        return t

    return t + b


def sigmoid(x: Variable) -> Variable:
    return 1 / (1 + exp(-x))


def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    return MeanSquaredError()(x0, x1)


def softmax1d(x: Variable) -> Variable:
    return softmax(x, None)


def softmax(x: Variable, axis: Optional[int] = 1) -> Variable:
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    n = x.shape[0]
    assert t.shape == (n,)

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)

    # example
    # log_p
    # array([[-1.55738458, -1.37022841, -0.62499391★],
    #        [-1.65968355★, -1.14549491, -0.70981659],
    #        [-1.53500807, -1.97749028★, -0.43675478],
    #        [-1.72480927★, -1.28536242, -0.6065244 ]])
    #
    # np.arange = (array([0, 1, 2, 3])
    # t.data = array([2, 0, 1, 0]))
    #
    # array([-0.62499391, -1.65968355, -1.97749028, -1.72480927])
    tlog_p = log_p[np.arange(n), t.data]
    return sum(tlog_p) / -n


def accuracy(y: Variable, t: Variable) -> Variable:
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()
    return y.new_variable(acc)


def relu(x: Variable) -> Variable:
    return ReLU()(x)
