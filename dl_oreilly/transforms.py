from typing import Any, Callable, Type, cast

import numpy as np

from . import NDFloatArray

Transform = Callable[[NDFloatArray], NDFloatArray]


class Compose:
    def __init__(self, transforms: list[Transform]) -> None:
        self._transforms = transforms

    def __call__(self, x: NDFloatArray) -> NDFloatArray:
        for t in self._transforms:
            x = t(x)
        return x


class Normalize:
    def __init__(self, mean: float | NDFloatArray = 0.0, std: float | NDFloatArray = 1.0) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, array: NDFloatArray) -> NDFloatArray:
        mean, std = self._mean, self._std

        if not np.isscalar(mean):
            mean = cast(NDFloatArray, mean)
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(mean) == 1 else len(mean)
            mean = np.array(mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            std = cast(NDFloatArray, std)
            mshape = [1] * array.ndim
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(std) == 1 else len(std)
            std = np.array(std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    def __call__(self, array: NDFloatArray) -> NDFloatArray:
        return array.flatten()


class AsType:
    def __init__(self, dtype: Type[Any]) -> None:
        self._dtype = dtype

    def __call__(self, array: NDFloatArray) -> NDFloatArray:
        return array.astype(self._dtype)


class ToFloat(AsType):
    def __init__(self, dtype: Type[np.float_] = np.float32) -> None:
        self.dtype = dtype


class ToInt(AsType):
    def __init__(self, dtype: Type[np.int_] = np.int_) -> None:
        self.dtype = dtype
