from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Type

import numpy as np

from .function import linear
from .protocol import Variable
from .variable import Parameter


class Layer(ABC):
    def __init__(self) -> None:
        self._params: set[str] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, x: Variable) -> Variable:
        output = self.forward(x)
        return output

    @abstractmethod
    def forward(self, x: Variable) -> Variable:
        ...

    def clear_grad(self) -> None:
        for param in self.params():
            param.clear_grad()

    def params(self) -> Iterator[Parameter]:
        for name in self._params:
            yield self.__dict__[name]


class Linear(Layer):
    def __init__(
        self,
        out_size: int,
        nobias: bool = False,
        dtype: Type[np.floating[Any]] = np.float32,
    ) -> None:
        super().__init__()

        self._out_size = out_size
        self._dtype = dtype

        self.W: Optional[Parameter] = None
        if nobias:
            self.b = None
        else:
            zero = np.zeros(out_size, dtype=dtype)
            self.b = Parameter(zero, name="b")

    def _init_w(self, in_size: int) -> None:
        w_data = np.random.randn(in_size, self._out_size).astype(self._dtype) * np.sqrt(1 / in_size)
        self.W = Parameter(w_data, name="W")

    def forward(self, x: Variable) -> Variable:
        if self.W is None:
            self._init_w(x.shape[1])
        assert self.W is not None

        return linear(x, self.W, self.b)
