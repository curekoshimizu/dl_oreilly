import heapq
from typing import Any

from . import NDFloatArray
from .protocol import Function, Variable


class _ComparableFunction:
    """
    used for Variable'backward implementation
    """

    def __init__(self, f: Function) -> None:
        self._f = f

    @property
    def generation(self) -> int:
        return self._f.generation

    @property
    def function(self) -> Function:
        return self._f

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, _ComparableFunction)
        return self.generation == other.generation

    def __lt__(self, other: Any) -> bool:
        """
        reverse order of generation
        """
        assert isinstance(other, _ComparableFunction)
        return self.generation > other.generation


class DummyFunction:
    def __init__(self, generation: int) -> None:
        self._generation = generation

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def inputs(self) -> tuple[Variable, ...]:
        raise NotImplementedError()

    @property
    def output(self) -> Variable:
        raise NotImplementedError()

    def backward(self, grad: NDFloatArray) -> tuple[NDFloatArray, ...]:
        raise NotImplementedError()


class _FunctionPriorityQueue:
    """
    used for Variable'backward implementation
    """

    def __init__(self) -> None:
        self._set: set[Function] = set()
        self._list: list[_ComparableFunction] = []

    def register(self, f: Function) -> bool:
        if f in self._set:
            # already registered
            return False

        self._set.add(f)
        heapq.heappush(self._list, _ComparableFunction(f))
        return True

    def pop(self) -> Function:
        ret = heapq.heappop(self._list)
        return ret.function

    def is_empty(self) -> bool:
        return len(self._list) == 0
