import numpy as np

from .variable import Var


def test_variable() -> None:
    x = Var(np.array([1.0, 2.0]), name="x")
    assert x.ndim == 1
    assert x.size == 2
    assert x.shape == (2,)
    assert x.dtype == np.dtype(np.float64)
    assert len(x) == 2
    assert repr(x) == "variable(x:[1. 2.])"
