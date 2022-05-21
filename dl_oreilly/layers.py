from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Type

import numpy as np

from . import NDFloatArray
from .function import linear
from .graph import Graphviz
from .protocol import Variable
from .variable import Parameter


class Layer(ABC):
    def __init__(self) -> None:
        self._params: set[str] = set()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, x: Variable) -> Variable:
        output = self.forward(x)
        return output

    def _flatten_params(self, params_dict: dict[str, Variable], parent_key: str = "") -> None:
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + "/" + name if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path: pathlib.Path) -> None:
        params_dict: dict[str, Variable] = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()}
        try:
            np.savez_compressed(str(path), **array_dict)
        except (Exception, KeyboardInterrupt):
            if path.exists():
                path.unlink()

    def load_weights(self, path: pathlib.Path) -> None:
        npz = np.load(str(path))
        params_dict: dict[str, Variable] = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

    @abstractmethod
    def forward(self, x: Variable) -> Variable:
        ...

    def clear_grad(self) -> None:
        for param in self.params():
            param.clear_grad()

    def params(self) -> Iterator[Parameter]:
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj


class Linear(Layer):
    def __init__(
        self,
        out_size: int,
        nobias: bool = False,
        dtype: Type[np.floating[Any]] = np.float32,
    ) -> None:
        super().__init__()

        self._out_size = out_size
        self._dtype = dtype

        self.W: Optional[Parameter] = None
        if nobias:
            self.b = None
        else:
            zero = np.zeros(out_size, dtype=self._dtype)
            self.b = Parameter(zero, name="b")

    def _init_w(self, in_size: int) -> None:
        w_data = np.random.randn(in_size, self._out_size).astype(self._dtype) * np.sqrt(1 / in_size)
        self.W = Parameter(w_data, name="W")

    def forward(self, x: Variable) -> Variable:
        if self.W is None:
            self._init_w(x.shape[1])
        assert self.W is not None

        return linear(x, self.W, self.b)


class Conv2d(Layer):
    def __init__(
        self,
        out_channels: int,
        *,
        kernel_size: int = 0,
        stride: int = 1,
        pad: int = 0,
        dtype: Type[np.floating[Any]] = np.float32,
    ) -> None:
        super().__init__()
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._pad = pad
        self._dtype = dtype

        self.W: Optional[Parameter] = None
        self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_w(self, in_channels: int) -> None:
        c, oc = in_channels, self._out_channels
        kh, kw = (self._kernel_size, self._kernel_size)
        scale = np.sqrt(1 / (c * kh * kw))
        w_data = np.random.randn(oc, c, kh, kw).astype(self._dtype) * scale
        self.W = Parameter(w_data, name="W")

    def forward(self, x: Variable) -> Variable:
        if self.W is None:
            self._init_w(x.shape[1])
        assert self.W is not None

        return _conv2d(x, self.W, self.b, self._stride, self._pad)


def _conv2d(x: Variable, W: Variable, b: Variable, stride: int, pad: int) -> Variable:
    kh, kw = W.shape[2:]
    col = _im2col_array(x.data, (kh, kw), stride, pad, to_matrix=False)

    y = np.tensordot(col.data, W.data, ((1, 2, 3), (1, 2, 3)))
    y += b.data
    y = np.rollaxis(y, 3, 1)
    return x.new_variable(y)


def _im2col_array(
    img: NDFloatArray, kernel_size: tuple[int, int], stride: int, pad: int, to_matrix: bool = True
) -> NDFloatArray:

    N, C, H, W = img.shape
    KH, KW = kernel_size
    SH, SW = (stride, stride)
    PH, PW = (pad, pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode="constant", constant_values=(0,))
    col: NDFloatArray = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def get_conv_outsize(input_size: int, kernel_size: int, stride: int, pad: int) -> int:
    return (input_size + pad * 2 - kernel_size) // stride + 1


class Model(Layer):
    def save_graph(self, x: Variable, path: Optional[pathlib.Path] = None) -> None:
        y = self.forward(x)
        g = Graphviz()

        if path is None:
            name = y.name
            if name is None:
                name = "variable"
            path = pathlib.Path(f"{name}.png")
        g.save(y, path)
