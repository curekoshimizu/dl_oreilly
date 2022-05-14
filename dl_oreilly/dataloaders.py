from __future__ import annotations

import math

import numpy as np

from .datasets import Dataset
from .protocol import Variable
from .variable import Var


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._data_size = len(dataset)
        self._max_iter = math.ceil(self._data_size / batch_size)
        self.reset()

    def reset(self) -> None:
        self._iteration = 0
        if self._shuffle:
            self._index = np.random.permutation(self._data_size)
        else:
            self._index = np.arange(self._data_size)

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> tuple[Variable, Variable]:
        if self._iteration >= self._max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self._iteration, self._batch_size

        batch_indexes = self._index[i * batch_size : (i + 1) * batch_size]

        batch = [self._dataset[i] for i in batch_indexes]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])
        self._iteration += 1
        return Var(x), Var(t)

    def next(self) -> tuple[Variable, Variable]:
        return next(self)
