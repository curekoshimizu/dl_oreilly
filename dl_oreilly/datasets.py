import gzip
import pathlib
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np

from . import NDFloatArray, NDIntArray
from .transforms import Compose, Flatten, Normalize, ToFloat, Transform


def identity(x: NDFloatArray) -> NDFloatArray:
    return x


class Dataset(ABC):
    def __init__(self, train: bool = True, transform: Optional[Transform] = None) -> None:
        self._train = train
        if transform is None:
            transform = identity
        self._transform = transform
        self._data, self._label = self._prepare()

    def __getitem__(self, index: int) -> tuple[NDFloatArray, int]:
        return self._transform(self._data[index]), self._label[index]

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


class MNIST(Dataset):
    def __init__(
        self,
        train: bool = True,
        transform: Optional[Transform] = None,
    ):
        if transform is None:
            transform = Compose([Flatten(), ToFloat(), Normalize(0.0, 255.0)])
        super().__init__(train, transform)

    def _prepare(self) -> tuple[NDFloatArray, NDIntArray]:
        url = "http://yann.lecun.com/exdb/mnist/"
        train_files = {"target": "train-images-idx3-ubyte.gz", "label": "train-labels-idx1-ubyte.gz"}
        test_files = {"target": "t10k-images-idx3-ubyte.gz", "label": "t10k-labels-idx1-ubyte.gz"}

        files = train_files if self._train else test_files
        data_path = get_file(url + files["target"])
        label_path = get_file(url + files["label"])

        data = self._load_data(str(data_path))
        label = self._load_label(str(label_path))
        return data, label

    def _load_label(self, filepath: str) -> NDIntArray:
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return cast(NDIntArray, labels)

    def _load_data(self, filepath: str) -> NDFloatArray:
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return cast(NDFloatArray, data)

    def show(self, row: int = 10, col: int = 10) -> None:
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H : (r + 1) * H, c * W : (c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)
                ].reshape(H, W)
        plt.imshow(img, cmap="gray", interpolation="nearest")  # type: ignore
        plt.axis("off")  # type: ignore
        plt.show()

    @staticmethod
    def labels() -> dict[int, str]:
        return {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}


_cache_dir = pathlib.Path.home() / ".dl"


def get_file(url: str, file_name: Optional[str] = None) -> pathlib.Path:
    if file_name is None:
        file_name = url[url.rfind("/") + 1 :]
    file_path = _cache_dir / file_name

    if not _cache_dir.exists():
        _cache_dir.mkdir()

    if file_path.exists():
        return file_path

    try:
        urllib.request.urlretrieve(url, file_path)
    except (Exception, KeyboardInterrupt):
        if file_path.exists():
            file_path.unlink()
        raise

    return file_path
