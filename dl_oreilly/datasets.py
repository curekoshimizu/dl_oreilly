from abc import ABC, abstractmethod

import numpy as np

from . import NDFloatArray, NDIntArray


class Dataset(ABC):
    def __init__(self, train: bool = True) -> None:
        self._train = train
        self._data, self._label = self._prepare()

    def __getitem__(self, index: int) -> tuple[float, int]:
        return self._data[index], self._label[index]

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> NDFloatArray:
        return self._data

    @property
    def label(self) -> NDIntArray:
        return self._label

    @abstractmethod
    def _prepare(self) -> tuple[NDFloatArray, NDIntArray]:
        ...


class Spiral(Dataset):
    def _prepare(self) -> tuple[NDFloatArray, NDIntArray]:
        seed = 1984 if self._train else 2020
        np.random.seed(seed)

        num_data, num_class, input_dim = 100, 3, 2
        data_size = num_class * num_data
        x = np.zeros((data_size, input_dim), dtype=np.float32)
        t = np.zeros(data_size, dtype=np.int_)

        for j in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
                ix = num_data * j + i
                x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
                t[ix] = j
        # Shuffle
        indices = np.random.permutation(num_data * num_class)
        x = x[indices]
        t = t[indices]
        return x, t
