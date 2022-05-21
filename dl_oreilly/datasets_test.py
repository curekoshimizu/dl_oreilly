import numpy as np
import matplotlib.pyplot as plt

from .datasets import MNIST

def _save_figure(x, save: bool) -> None:
    if not save:
        return

    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.savefig("mnist.png")


def test_mnist_data(save: bool = False) -> None:
    train_set = MNIST(train=True)
    test_set = MNIST(train=False)
    assert len(train_set) == 60000
    assert len(test_set) == 10000

    x, t = train_set[0]
    assert isinstance(x, (np.ndarray, np.generic))
    assert t == 5

    _save_figure(x, save)
