import numpy as np
from PIL import Image

from .config import use_test_mode
from .datasets import get_file
from .models import VGG16
from .variable import Var


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
    with use_test_mode():
        y = model(Var(x))

    assert np.argmax(y.data) == 198
