from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import numpy as np

from . import NDFloatArray


class Variable:
    def __init__(self, data: NDFloatArray, name: Optional[str] = None) -> None:
        self._data = data
        self._creator: Optional[Function] = None
        self._grad: Optional[NDFloatArray] = None
        self._name = name
        self._generation = 0

    def backward(self) -> None:
        self.grad = np.ones_like(self.data)
        variables: list[Variable] = [self]

        # funcs: list[ Function] = []
        # f = self.creator
        # if f is not None:
        #     funcs.append(f)
        #
        #
        # while len(funcs) > 0:
        #     funcs

        while len(variables) > 0:
            # print("-[status]---------------")
            # for x in variables:
            #     print(x.name, "grad =", x.grad)
            # print("------------------------")

            variable = variables.pop(0)
            f = variable.creator
            if f is None:
                continue

            xs = f.inputs
            grads: tuple[NDFloatArray, ...]
            grads = f.backward(variable.grad)
            assert len(xs) == len(grads)
            for x, grad in zip(xs, grads):
                base = np.array(0.0) if x._grad is None else x._grad
                x._grad = grad + base
                # print(x.name, "updated", x._grad)
            variables.extend(xs)

    @property
    def grad(self) -> NDFloatArray:
        assert self._grad is not None, "grad is not computed."
        return self._grad

    @grad.setter
    def grad(self, grad: NDFloatArray) -> None:
        self._grad = grad

    def clear_grad(self) -> None:
        self._grad = None

    def set_creator(self, f: Function) -> None:
        self._creator = f
        self._generation = f.generation + 1

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def data(self) -> NDFloatArray:
        return self._data

    @property
    def creator(self) -> Optional[Function]:
        return self._creator

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


class ComparableFunction:
    def __init__(self, f: Function) -> None:
        self._f = f

    @property
    def generation(self) -> int:
        return self._f.generation

    @property
    def function(self) -> Function:
        return self._f

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, ComparableFunction)
        return self.generation == other.generation

    def __lt__(self, other: Any) -> bool:
        """
        reverse order of generation
        """
        assert isinstance(other, ComparableFunction)
        return self.generation > other.generation


class Function(Protocol):
    @property
    def generation(self) -> int:
        ...

    @property
    def inputs(self) -> tuple[Variable, ...]:
        ...

    @abstractmethod
    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        ...


class DummyFunction:
    def __init__(self, generation: int) -> None:
        self._generation = generation

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def inputs(self) -> tuple[Variable, ...]:
        raise NotImplementedError()

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        raise NotImplementedError()


class _FunctionPriorityQueue:
    """
    used for Variable'backward implementation
    """

    def __init__(self) -> None:
        self._set: set[Function] = set()
        self._list: list[ComparableFunction] = []

    def register(self, f: Function) -> bool:
        if f in self._set:
            # already registered
            return False

        self._set.add(f)
        heapq.heappush(self._list, ComparableFunction(f))
        return True

    def pop(self) -> Function:
        ret = heapq.heappop(self._list)
        return ret.function

    def is_empty(self) -> bool:
        return len(self._list) == 0


class OneArgFunction(ABC):
    def __call__(self, input: Variable) -> Variable:
        self._input = input
        assert getattr(self, "_generation", None) is None, "this function has already been called. but called again!"
        self._generation = input.generation
        y = self.forward(input.data)

        output = Variable(y)
        output.set_creator(self)

        self._output = output
        return output

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def input(self) -> Variable:
        return self._input

    @property
    def inputs(self) -> tuple[Variable, ...]:
        return (self._input,)

    @property
    def x(self) -> NDFloatArray:
        return self._input.data

    @abstractmethod
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        ...

    @abstractmethod
    def _backward_core(self, x: NDFloatArray) -> NDFloatArray:
        ...

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        return (self._backward_core(grad),)


class TwoArgsFunction(ABC):
    def __call__(self, x1: Variable, x2: Variable) -> Variable:
        self._inputs = (x1, x2)
        assert getattr(self, "_generation", None) is None, "this function has already been called. but called again!"
        self._generation = max(x1.generation, x2.generation)
        y = self.forward(x1.data, x2.data)

        output = Variable(y)
        output.set_creator(self)

        self._output = output
        return output

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def inputs(self) -> tuple[Variable, Variable]:
        return self._inputs

    @property
    def xs(self) -> tuple[NDFloatArray, NDFloatArray]:
        return (self._inputs[0].data, self._inputs[1].data)

    @abstractmethod
    def forward(self, x: NDFloatArray, y: NDFloatArray) -> NDFloatArray:
        ...

    @abstractmethod
    def _backward_core(self, grad: NDFloatArray) -> tuple[NDFloatArray, NDFloatArray]:
        ...

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        return self._backward_core(grad)
