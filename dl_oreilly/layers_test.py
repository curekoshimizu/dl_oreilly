from .layers import get_conv_outsize


def test_get_conv_outsize() -> None:
    assert get_conv_outsize(input_size=4, kernel_size=3, stride=1, pad=0) == 2
    assert get_conv_outsize(input_size=4, kernel_size=3, stride=1, pad=1) == 4
    assert get_conv_outsize(input_size=7, kernel_size=3, stride=2, pad=0) == 3
