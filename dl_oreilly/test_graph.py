import pathlib
import tempfile

import numpy as np

from .graph import Graphviz
from .variable import Var


def test_dot_var() -> None:
    g = Graphviz()
    x = Var(np.array([2.0]), name="x")
    text = g._dot_var(x)
    start = text.find("[")
    end = text.rfind("]")

    assert text[start + 1 : end] == "label=\"x:[2.]\", color=orange, style=filled"


def test_dot_func() -> None:
    g = Graphviz()
    x = Var(np.array([2.0]), name="x")
    y = x + x
    f = y.creator
    assert f is not None
    text = g._dot_func(f)
    start = text.find("[")
    end = text.rfind("]")

    assert text[start + 1 : end] == "label=\"add\", color=lightblue, style=filled, shape=box"


def test_save() -> None:
    g = Graphviz()
    x = Var(np.array([2.0]), name="x")
    y = x + 2 * x
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "hoge.png"
        assert not f.exists()
        g.save(y, f)
        assert f.exists()
