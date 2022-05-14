import matplotlib.pyplot as plt
import numpy as np

from . import NDFloatArray


def _save_figure(x: NDFloatArray, y: NDFloatArray, save: bool) -> None:
    if not save:
        return

    plt.figure(figsize=(10, 7.5))
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig("plot.png")


def test_simple_nn(save: bool = False) -> None:
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    _save_figure(x, y, save)
