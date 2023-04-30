import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .basemodule import Module
from .variable import AGarray, Param

__all__ = ["Conv1d", "Conv2d"]


class Conv1d(Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        pad: int = 0,
        bias: bool = True,
    ):
        super(Conv1d, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        # TODO: 待确认
        self.weight = Param(
            np.random.randn(out_chs, in_chs, kernel_size, kernel_size) * 0.01
        )
        self._parameters["weight"] = self.weight
        if bias:
            self.bias = Param(np.random.randn(out_chs) * 0.01)  # bias是一维数组
            self._parameters["bias"] = self.bias

    def forward(self, x):
        assert x.ndim == 3, "输入x不是3维的！"
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}: in_chs={self.in_chs}, out_chs={self.out_chs}, kernel_size={self.kernel_size}, stride={self.stride}, pad={self.pad}\n"


class Conv2d(Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        pad: int = 0,
        bias: bool = True,
    ):
        super(Conv2d, self).__init__()
        assert kernel_size - pad - 1 >= 0, "padding太大了!"
        self.in_chs = int(in_chs)
        self.out_chs = int(out_chs)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.pad = int(pad)
        self.weight = Param(
            np.random.randn(out_chs, in_chs, kernel_size, kernel_size) * 0.01
        )
        self._parameters["weight"] = self.weight
        if bias:
            self.bias = Param(np.random.randn(out_chs) * 0.01)  # bias是一维数组
            self._parameters["bias"] = self.bias

    def forward(self, x):
        # assert x.ndim == 4, "输入x不是四维的！"
        # self.input = x
        # N, in_chs, H, W = x.shape
        # K = self.kernel_size
        # pad = self.pad
        # stride = self.stride
        # out_chs = self.out_chs

        # TODO:
        raise NotImplementedError

    def backward(self, delta):
        """delta和self.output形状相同，N * out_chs * H_o * W_o"""
        # N, in_chs, H, W = self.input.shape
        # N, out_chs, H_o, W_o = delta.shape
        # K = self.kernel_size
        # pad = self.pad
        # stride = self.stride

        # TODO:
        raise NotImplementedError

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}: in_chs={self.in_chs}, out_chs={self.out_chs}, kernel_size={self.kernel_size}, stride={self.stride}, pad={self.pad}\n"
