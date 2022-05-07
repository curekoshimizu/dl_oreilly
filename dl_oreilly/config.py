import contextlib
from typing import Iterator


class Config:
    enable_backprop = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    old = Config.enable_backprop
    Config.enable_backprop = False
    try:
        yield
    finally:
        Config.enable_backprop = old
