from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import NDFloatArray
from .layers import Layer
from .protocol import Variable


class Optimizer(ABC):
    def __init__(self) -> None:
        self._target: Optional[Layer] = None
        # self._hooks = []

    def setup(self, target: Layer) -> Optimizer:
        self._target = target
        return self

    def update(self) -> None:
        if self._target is None:
            return

        params: list[Variable] = [p for p in self._target.params() if p.grad is not None]

        # for f in self._hooks:
        #     f(params)

        for param in params:
            self._update_one(param)

    @abstractmethod
    def _update_one(self, param: Variable) -> None:
        ...

    # def add_hook(self, f) -> None:
    #     self._hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self._lr = lr

    def _update_one(self, param: Variable) -> None:
        param.data -= self._lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self._lr = lr
        self._momentum = momentum
        self._vs: dict[int, NDFloatArray] = {}

    def _update_one(self, param: Variable) -> None:

        v_key = id(param)
        if v_key not in self._vs:
            self._vs[v_key] = np.zeros_like(param.data)

        v = self._vs[v_key]
        v *= self._momentum
        param.data -= self._lr * param.grad.data
        param.data += v
