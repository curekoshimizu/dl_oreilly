import matplotlib.pyplot as plt
import numpy as np

from .dataloaders import DataLoader
from .datasets import Spiral
from .function import softmax_cross_entropy
from .models import MLP
from .optimizers import SGD
from .variable import Var


def test_spiral(save: bool = True) -> None:
    max_epoch = 300
    batch_size = 30

    train_set = Spiral(train=True)
    # test_set = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size, shuffle=False)

    np.random.seed(0)
    model = MLP((10, 3))
    optimizer = SGD(lr=1.0).setup(model)

    data_size = len(train_set)

    trace_loss = []
    for epoch in range(max_epoch):
        sum_loss = 0.0

        for batch_x, batch_t in train_loader:
            y = model(batch_x)
            loss = softmax_cross_entropy(y, batch_t)
            model.clear_grad()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        ave_loss = sum_loss / data_size
        trace_loss.append(ave_loss)

    if save:
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(trace_loss)), trace_loss, label="train")
        plt.xlabel("iterations (epoch)")
        plt.ylabel("loss")
        plt.title("Cross Entropy Loss", fontsize=20)
        plt.grid()
        plt.savefig("loss.png")

    if save:
        x0_line = np.arange(-1.1, 1.1, 0.005)
        x1_line = np.arange(-1.1, 1.1, 0.005)
        x0_grid, x1_grid = np.meshgrid(x0_line, x1_line)
        x_point = Var(np.c_[x0_grid.ravel(), x1_grid.ravel()])
        y = model(x_point)
        predict_cls = np.argmax(y.data, axis=1)
        y_grid = predict_cls.reshape(x0_grid.shape)

        x = train_set.data
        t = train_set.label
        markers = ["o", "x", "^"]

        plt.figure(figsize=(8, 8))
        plt.contourf(x0_grid, x1_grid, y_grid)  # type:ignore
        for i in range(3):
            plt.scatter(x[t == i, 0], x[t == i, 1], marker=markers[i], s=50, label="class " + str(i))
        plt.xlabel("$x_0$", fontsize=15)
        plt.ylabel("$x_1$", fontsize=15)
        plt.title("iter:" + str(max_epoch) + ", loss=" + str(np.round(loss.data, 5)) + ", N=" + str(len(x)), loc="left")
        plt.legend()
        plt.savefig("spiral.png")
