from typing import Callable, Optional

import numpy as np

from .datasets import get_file
from .function import dropout, pooling, relu, reshape, sigmoid
from .layers import Conv2d, Layer, Linear, Model
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


WEIGHTS_PATH = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"


class VGG16(Model):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = Linear(4096)
        self.fc7 = Linear(4096)
        self.fc8 = Linear(1000)

        weights_path = get_file(WEIGHTS_PATH)
        self.load_weights(weights_path)

    def forward(self, x: Variable) -> Variable:
        x = relu(self.conv1_1(x))
        x = relu(self.conv1_2(x))
        x = pooling(x, 2, 2)
        x = relu(self.conv2_1(x))
        x = relu(self.conv2_2(x))
        x = pooling(x, 2, 2)
        x = relu(self.conv3_1(x))
        x = relu(self.conv3_2(x))
        x = relu(self.conv3_3(x))
        x = pooling(x, 2, 2)
        x = relu(self.conv4_1(x))
        x = relu(self.conv4_2(x))
        x = relu(self.conv4_3(x))
        x = pooling(x, 2, 2)
        x = relu(self.conv5_1(x))
        x = relu(self.conv5_2(x))
        x = relu(self.conv5_3(x))
        x = pooling(x, 2, 2)
        x = reshape(x, (x.shape[0], -1))
        x = dropout(relu(self.fc6(x)))
        x = dropout(relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert("RGB")
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image
