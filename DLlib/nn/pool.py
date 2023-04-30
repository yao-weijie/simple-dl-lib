import numpy as np

from .basemodule import Module
from .functional.img2col import img2col_1d, img2col_2d

__all__ = [
    "MaxPool1d",
    "MaxPool2d",
    "MeanPool1d",
    "MeanPool2d",
    "AvgPool1d",
    "AvgPool2d",
]


class MaxPool1d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, pad: int = 0):
        super(MaxPool1d, self).__init__()
        pass

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, pad: int = 0):
        super(MaxPool2d, self).__init__()
        assert kernel_size - pad - 1 >= 0, "padding太大了"
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class MeanPool1d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, pad: int = 0):
        super(AvgPool2d, self).__init__()
        pass

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class MeanPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, pad: int = 0):
        super(MeanPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class AvgPool1d(Module):
    def __init__(self):
        super(AvgPool1d, self).__init__()

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError


class AvgPool2d(Module):
    def __init__(self):
        super(AvgPool2d, self).__init__()

    def forward(self, x):
        """shape of x: N*C*H*W"""
        self.input = x
        self.output = np.mean(x, axis=(2, 3))  # N * C

        return self.output

    def backward(self, delta):
        N, C, H, W = self.input.shape
        delta /= H * W
        delta = delta.reshape(N, C, 1, 1)
        self.grad_input = np.ones_like(self.input) * delta

        return self.grad_input
