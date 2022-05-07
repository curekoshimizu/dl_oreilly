from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import NDFloatArray
from .backward_helper import _FunctionPriorityQueue
from .function import mul
from .protocol import Function, Variable


class Var:
    def __init__(self, data: NDFloatArray, name: Optional[str] = None) -> None:
        self._data = data
        self._creator: Optional[Function] = None
        self._grad: Optional[NDFloatArray] = None
        self._name = name
        self._generation = 0

    def new_variable(cls, data: NDFloatArray, name: Optional[str] = None) -> Variable:
        return Var(data, name=name)

    def backward(self) -> None:
        self.grad = np.ones_like(self.data)
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
                x.grad = grad + base
                f = x.creator
                if f is not None:
                    queue.register(f)

    @property
    def optional_grad(self) -> Optional[NDFloatArray]:
        return self._grad

    @property
    def grad(self) -> NDFloatArray:
        assert self._grad is not None, "grad is not computed."
        return self._grad

    @grad.setter
    def grad(self, grad: NDFloatArray) -> None:
        self._grad = grad

    def clear_grad(self) -> None:
        self._grad = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def data(self) -> NDFloatArray:
        return self._data

    @property
    def creator(self) -> Optional[Function]:
        return self._creator

    @creator.setter
    def creator(self, f: Function) -> None:
        self._creator = f
        self._generation = f.generation + 1

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        name = "" if self.name is None else f"{self.name}:"
        return f"variable({name}{self.data})"
