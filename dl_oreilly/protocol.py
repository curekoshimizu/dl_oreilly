from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import numpy as np

from . import NDFloatArray


class Function(Protocol):
    @property
    def generation(self) -> int:
        ...

    @property
    def inputs(self) -> tuple[Variable, ...]:
        ...

    @property
    def output(self) -> Variable:
        ...

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        ...


class Variable(ABC):
    def __init__(self, data: NDFloatArray, name: Optional[str]) -> None:
        self._data = data
        self._name = name
        self._grad: Optional[NDFloatArray] = None
        self._creator: Optional[Function] = None
        self._generation = 0

    @property
    def optional_grad(self) -> Optional[NDFloatArray]:
        return self._grad

    @property
    def grad(self) -> NDFloatArray:
        assert self._grad is not None, "grad is not computed."
        return self._grad

    def _set_grad(self, grad: NDFloatArray) -> None:
        self._grad = grad

    def clear_grad(self) -> None:
        self._grad = None

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name
        ...

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
    def generation(self) -> int:
        return self._generation

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

    @abstractmethod
    def new_variable(cls, data: NDFloatArray) -> Variable:
        ...

    @abstractmethod
    def backward(self) -> None:
        ...

    @abstractmethod
    def __mul__(self, other: Variable | NDFloatArray) -> Variable:
        ...

    @abstractmethod
    def __add__(self, other: Variable | NDFloatArray) -> Variable:
        ...

    @abstractmethod
    def __rmul__(self, other: NDFloatArray) -> Variable:
        ...

    @abstractmethod
    def __radd__(self, other: NDFloatArray) -> Variable:
        ...
