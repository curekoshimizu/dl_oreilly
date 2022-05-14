from typing import Callable, Optional

from .function import sigmoid
from .layers import Layer, Linear, Model
from .protocol import Variable


class TwoLayerNet(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = Linear(out_size=hidden_size)
        self.l2 = Linear(out_size=out_size)

    def forward(self, x: Variable) -> Variable:
        y = sigmoid(self.l1(x))
        return self.l2(y)


class MLP(Model):
    """
    MLP = Multi-Layer Perceptron
    """

    def __init__(
        self, fc_output_sizes: tuple[int, ...], activation: Optional[Callable[[Variable], Variable]] = None
    ) -> None:
        super().__init__()
        self._layers: list[Layer] = []

        if activation is None:
            activation = sigmoid
        self._activation = activation

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self._layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        # applay except the last layer
        for layer in self._layers[:-1]:
            x = self._activation(layer(x))
        # apply last layer
        return self._layers[-1](x)
