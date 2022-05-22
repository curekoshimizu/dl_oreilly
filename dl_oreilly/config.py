import contextlib
from typing import Iterator


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    with enable_backprop(False):
        yield


@contextlib.contextmanager
def enable_backprop(flag: bool) -> Iterator[None]:
    old = Config.enable_backprop
    Config.enable_backprop = flag
    try:
        yield
    finally:
        Config.enable_backprop = old


@contextlib.contextmanager
def use_test_mode() -> Iterator[None]:
    old = Config.train
    Config.train = False
    try:
        yield
    finally:
        Config.train = old
