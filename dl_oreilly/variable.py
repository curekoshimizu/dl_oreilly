from __future__ import annotations

import pathlib
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import NDFloatArray
from .backward_helper import _FunctionPriorityQueue
from .config import enable_backprop
from .function import add, div, get_item, mul, neg, pow, reshape, sub, sum, transpose
from .graph import Graphviz
from .protocol import Variable


class Var(Variable):
    def __init__(self, data: NDFloatArray, name: Optional[str] = None) -> None:
        super().__init__(data, name)

    def new_variable(cls, data: NDFloatArray, name: Optional[str] = None) -> Variable:
        return Var(data, name=name)

    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
        with enable_backprop(create_graph):
            self._set_grad(Var(np.ones_like(self.data)))
            queue = _FunctionPriorityQueue()

            f0 = self.creator
            if f0 is not None:
                queue.register(f0)
            while not queue.is_empty():
                f = queue.pop()
                xs = f.inputs
                grads = f.backward(f.output.grad)
                assert len(xs) == len(grads)
                for x, grad in zip(xs, grads):
                    pre_grad = x.optional_grad
                    if pre_grad is None:
                        assert x.shape == grad.shape, (x.shape, grad.shape)
                        x._set_grad(grad)
                    else:
                        x._set_grad(grad + pre_grad)
                    f0 = x.creator
                    if f0 is not None:
                        queue.register(f0)
                if not retain_grad:
                    y = f.output
                    y._set_grad(None)

    def save_graph(self, path: Optional[pathlib.Path] = None) -> None:
        if path is None:
            name = self.name
            if name is None:
                name = "variable"
            path = pathlib.Path(f"{name}.png")
        g = Graphviz()
        g.save(self, path)

    def reshape(self, shape: tuple[int, ...]) -> Variable:
        return reshape(self, shape)

    def transpose(self) -> Variable:
        return transpose(self)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Variable:
        return sum(self, axis, keepdims)

    def __neg__(self) -> Variable:
        return neg(self)

    def __add__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        if not isinstance(other, Variable):
            other = Var(other)
        return add(self, other)

    def __sub__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        if not isinstance(other, Variable):
            other = Var(other)
        return sub(self, other)

    def __mul__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        if not isinstance(other, Variable):
            other = Var(other)
        return mul(self, other)

    def __truediv__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        if not isinstance(other, Variable):
            other = Var(other)
        return div(self, other)

    def __pow__(self, exp: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(exp, int) or isinstance(exp, float):
            exp = np.array([exp])
        if not isinstance(exp, Variable):
            exp = Var(exp)
        return pow(self, exp)

    def __radd__(self, other: NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        return add(Var(other), self)

    def __rsub__(self, other: NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        return sub(Var(other), self)

    def __rmul__(self, other: NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        return mul(Var(other), self)

    def __rtruediv__(self, other: NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array([other])
        return div(Var(other), self)

    def __getitem__(self, slices: int | tuple[NDArray[Any], ...]) -> Variable:
        return get_item(self, slices)


class Parameter(Var):
    pass


class LazyParameter(Parameter):
    def __init__(self, name: str) -> None:
        self._initialized = False
        self._name = name
        dummy_data = np.array([1])
        super().__init__(dummy_data, self._name)
        setattr(self, "_data", None)  # dummy data is overwritten by None (to avoid mypy)

    @property
    def initialized(self) -> bool:
        return self._initialized or self.data is not None

    def initialize(self, data: NDFloatArray) -> None:
        assert not self.initialized
        super().__init__(data, self._name)
