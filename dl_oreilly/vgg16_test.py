import numpy as np
import pytest
from PIL import Image

from .config import no_grad, use_test_mode
from .datasets import ImageNet, get_file
from .models import VGG16
from .variable import Var


@pytest.mark.heavy
def test_vgg16() -> None:
    url = "https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
    img_path = get_file(url)
    img: Image.Image
    with Image.open(img_path) as img:
        x = VGG16.preprocess(img)
    assert x.shape == (3, 224, 224)
    x = x[np.newaxis]
    assert x.shape == (1, 3, 224, 224)

    model = VGG16()
    with use_test_mode(), no_grad():
        y = model(Var(x))

    index = int(np.argmax(y.data))
    labels = ImageNet.labels()
    assert len(labels) == 1000
    assert index == 340
    assert labels[index] == "zebra"
