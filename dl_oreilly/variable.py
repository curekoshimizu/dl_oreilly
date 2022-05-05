from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import NDFloatArray


class Variable:
    def __init__(self, data: NDFloatArray) -> None:
        self._data = data
        self._creator: Optional[Function] = None

    def backward(self, grad: Optional[NDFloatArray] = None) -> NDFloatArray:
        if grad is None:
            grad = np.array([1.0])
        f = self.creator
        if f is not None:
            x: Variable = f.input
            grad_x: NDFloatArray = f.backward(grad)
            return x.backward(grad_x)
        return grad

    def set_creator(self, f: Function) -> None:
        self._creator = f

    @property
    def data(self) -> NDFloatArray:
        return self._data

    @property
    def creator(self) -> Optional[Function]:
        return self._creator


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        self._input = input
        x = input.data
        y = self.forward(x)

        output = Variable(y)
        output.set_creator(self)

        self._output = output
        return output

    @property
    def input(self) -> Variable:
        return self._input

    @property
    def x(self) -> NDFloatArray:
        return self._input.data

    @abstractmethod
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        ...

    @abstractmethod
    def backward(self, x: NDFloatArray) -> NDFloatArray:
        ...
