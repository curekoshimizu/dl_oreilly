from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

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

    @property
    def name(self) -> str:
        ...

    def backward(self, grad: Variable) -> tuple[Variable, ...]:
        ...


class Variable(ABC):
    def __init__(self, data: NDFloatArray, name: Optional[str]) -> None:
        self._data = data
        self._name = name
        self._grad: Optional[Variable] = None
        self._creator: Optional[Function] = None
        self._generation = 0

    @abstractmethod
    def save_graph(self, path: Optional[pathlib.Path] = None) -> None:
        ...

    @property
    def optional_grad(self) -> Optional[Variable]:
        return self._grad

    @property
    def grad(self) -> Variable:
        assert self._grad is not None, "grad is not computed or not retained."
        return self._grad

    def _set_grad(self, grad: Optional[Variable]) -> None:
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

    @data.setter
    def data(self, data: NDFloatArray) -> None:
        self._data = data

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
        ret = f"variable({name}{self.data})"
        ret = ret.replace("\n", "")
        return ret

    @abstractmethod
    def new_variable(cls, data: NDFloatArray) -> Variable:
        ...

    @abstractmethod
    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
        ...

    @abstractmethod
    def reshape(self, reshape: tuple[int, ...]) -> Variable:
        ...

    @abstractmethod
    def transpose(self) -> Variable:
        ...

    @property
    def T(self) -> Variable:
        return self.transpose()

    @abstractmethod
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Variable:
        ...

    @abstractmethod
    def __neg__(self) -> Variable:
        ...

    @abstractmethod
    def __add__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __sub__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __mul__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __pow__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __truediv__(self, other: Variable | NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __radd__(self, other: NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __rsub__(self, other: NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __rmul__(self, other: NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __rtruediv__(self, other: NDFloatArray | float | int) -> Variable:
        ...

    @abstractmethod
    def __getitem__(self, slices: int | tuple[NDArray[Any], ...]) -> Variable:
        ...
