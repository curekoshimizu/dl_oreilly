from typing import Any

from .variable import Parameter


class Layer:
    def __init__(self) -> None:
        self._params: set[str] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)
