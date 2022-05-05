from abc import ABC, abstractmethod
from typing import Any

from .variable import Variable


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        return Variable(y)

    @abstractmethod
    def forward(self, x: Any) -> Any:
        ...


class Square(Function):
    def forward(self, x: Any) -> Any:
        return x**2
