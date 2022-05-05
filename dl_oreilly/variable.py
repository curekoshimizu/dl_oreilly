from . import NDFloatArray


class Variable:
    def __init__(self, data: NDFloatArray) -> None:
        self._data = data

    @property
    def data(self) -> NDFloatArray:
        return self._data
