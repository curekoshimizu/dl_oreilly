from __future__ import annotations

from typing import Optional, Protocol

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


class Variable(Protocol):
    def new_variable(cls, data: NDFloatArray) -> Variable:
        ...

    def backward(self) -> None:
        ...

    @property
    def optional_grad(self) -> Optional[NDFloatArray]:
        ...

    @property
    def grad(self) -> NDFloatArray:
        ...

    @grad.setter
    def grad(self, grad: NDFloatArray) -> None:
        ...

    def clear_grad(self) -> None:
        ...

    @property
    def name(self) -> Optional[str]:
        ...

    @name.setter
    def name(self, name: str) -> None:
        ...

    @property
    def data(self) -> NDFloatArray:
        ...

    @property
    def creator(self) -> Optional[Function]:
        ...

    @creator.setter
    def creator(self, f: Function) -> None:
        ...

    @property
    def generation(self) -> int:
        ...

    def __mul__(self, other: Variable) -> Variable:
        ...

    def __add__(self, other: Variable) -> Variable:
        ...
