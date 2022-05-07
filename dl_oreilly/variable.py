from __future__ import annotations

from typing import Optional

import numpy as np

from . import NDFloatArray
from .backward_helper import _FunctionPriorityQueue
from .function import add, mul
from .protocol import Variable


class Var(Variable):
    def __init__(self, data: NDFloatArray, name: Optional[str] = None) -> None:
        super().__init__(data, name)

    def new_variable(cls, data: NDFloatArray, name: Optional[str] = None) -> Variable:
        return Var(data, name=name)

    def backward(self) -> None:
        self._set_grad(np.ones_like(self.data))
        queue = _FunctionPriorityQueue()

        f = self.creator
        if f is not None:
            queue.register(f)
        while not queue.is_empty():
            f = queue.pop()
            xs = f.inputs
            grads = f.backward(f.output.grad)
            assert len(xs) == len(grads)
            for x, grad in zip(xs, grads):
                pre_grad = x.optional_grad
                base = np.array(0.0) if pre_grad is None else pre_grad
                x._set_grad(grad + base)
                f = x.creator
                if f is not None:
                    queue.register(f)

    def __mul__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array(other)
        if not isinstance(other, Variable):
            other = Var(other)
        return mul(self, other)

    def __add__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array(other)
        if not isinstance(other, Variable):
            other = Var(other)
        return add(self, other)

    def __rmul__(self, other: NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array(other)
        return mul(self, Var(other))

    def __radd__(self, other: NDFloatArray | float | int) -> Variable:
        if isinstance(other, int) or isinstance(other, float):
            other = np.array(other)
        return add(self, Var(other))
