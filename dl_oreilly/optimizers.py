from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
