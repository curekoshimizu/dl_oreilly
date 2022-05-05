from typing import Any


class Variable:
    def __init__(self, data: Any) -> None:
        self._data = data

    @property
    def data(self) -> Any:
        return self._data
