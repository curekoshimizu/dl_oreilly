from abc import ABC, abstractmethod

import numpy as np

from . import NDFloatArray
from .variable import Variable


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        return Variable(y)

    @abstractmethod
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        ...


class Square(Function):
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return x**2


class Exp(Function):
    def forward(self, x: NDFloatArray) -> NDFloatArray:
        return np.exp(x)
