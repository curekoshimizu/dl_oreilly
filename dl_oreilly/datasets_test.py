import matplotlib.pyplot as plt
import numpy as np

from . import NDFloatArray
from .config import no_grad
from .dataloaders import DataLoader
from .datasets import MNIST
from .function import accuracy, softmax_cross_entropy
from .models import MLP
from .optimizers import SGD


def _save_figure(x: NDFloatArray, save: bool) -> None:
    if not save:
        return

    plt.imshow(x.reshape(28, 28), cmap="gray")  # type:ignore
    plt.axis("off")  # type:ignore
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


def test_mnist_network() -> None:
    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_set = MNIST(train=True)
    test_set = MNIST(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    np.random.seed(0)
    model = MLP((hidden_size, 10))
    optimizer = SGD().setup(model)

    for epoch in range(max_epoch):
        sum_loss = 0.0
        sum_acc = 0.0
        for x, t in train_loader:
            assert len(x) == batch_size
            assert len(t) == batch_size
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            model.clear_grad()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(f"train_loss : {sum_loss/len(train_set):.4f}, accuracy: {sum_acc/len(train_set):.4f}")

        sum_loss = 0.0
        sum_acc = 0.0
        with no_grad():
            for x, t in test_loader:
                assert len(x) == batch_size
                assert len(t) == batch_size
                y = model(x)
                loss = softmax_cross_entropy(y, t)
                acc = accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
        print(f"test_loss : {sum_loss/len(test_set):.4f}, accuracy: {sum_acc/len(test_set):.4f}")

    assert sum_acc > 0.85
