from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import NDFloatArray


class Variable:
    def __init__(self, data: NDFloatArray) -> None:
        self._data = data
        self._creator: Optional[Function] = None

    def backward(self) -> NDFloatArray:
        grad = np.ones_like(self.data)
        variables: list[Variable] = [self]
        while len(variables) > 0:
            variable = variables.pop(0)
            f = variable.creator
            if f is None:
                break
            variables.extend(f.inputs)
            grad = f.backward(grad)

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
    def __call__(self, *inputs: Variable) -> Variable:
        self._inputs: tuple[Variable, ...] = inputs
        xs: tuple[NDFloatArray, ...] = tuple(input.data for input in inputs)
        ys: tuple[NDFloatArray, ...] = self.forward(xs)

        output = Variable(*ys)
        output.set_creator(self)

        self._output = output
        return output

    @property
    def inputs(self) -> tuple[Variable, ...]:
        return self._inputs

    @property
    def x(self) -> NDFloatArray:
        assert len(self._inputs) == 1
        return self._inputs[0].data

    @abstractmethod
    def forward(self, x: tuple[NDFloatArray, ...]) -> tuple[NDFloatArray, ...]:
        ...

    @abstractmethod
    def backward(self, x: NDFloatArray) -> NDFloatArray:
        ...
