from typing import Sequence

from numpy.lib.stride_tricks import sliding_window_view

from ..variable import AGarray


def img2col_1d(x: AGarray, kernel_size: Sequence[int]):
    assert x.ndim == 3, "ndim error!"
    return sliding_window_view(x, kernel_size, axis=(-1,), writeable=False)


def img2col_2d(x: AGarray, kernel_size: Sequence[int]):
    assert x.ndim == 4, "ndim error!"
    return sliding_window_view(x, kernel_size, axis=(-2, -1), writeable=False)


def img2col_3d(x: AGarray, kernel_size: Sequence[int]):
    assert x.ndim == 5, "ndim error!"
    return sliding_window_view(x, kernel_size, axis=(-3, -2, -1), writeable=False)
