import numpy as np
from scipy import signal
from .basemodule import Module
from .variable import Param


__all__ = ['Conv2d']


def rotate180(w):
    """将特征图旋转180度"""
    return np.rot90(w, k=2, axes=(-2, -1))


def fftconvolve(x, w, rotate=False):
    """
    x, w都是三维的
    signal.fftconvolve这个函数中会先将w旋转，至于深度学习中不用旋转，
    那就先旋转一次，再计算的时候就会正过来
    """
    if rotate is False:
        w = rotate180(w)
    return signal.fftconvolve(x, w, mode='valid', axes=(-2, -1))


class Conv2d(Module):
    def __init__(self, in_chs: int, out_chs: int, kernel_size: int,
                        stride: int = 1, pad: int = 0, bias: bool = True):
        super(Conv2d, self).__init__()
        assert kernel_size - pad -1 >= 0, 'padding太大了!'
        self.in_chs = int(in_chs)
        self.out_chs = int(out_chs)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.pad = int(pad)
        self.weight = Param(np.random.randn(out_chs, in_chs, kernel_size, kernel_size) * 0.01)
        self._parameters['weight'] = self.weight
        if bias:
            self.bias = Param(np.random.randn(out_chs) * 0.01)  # bias是一维数组
            self._parameters['bias'] = self.bias

    def forward(self, x):
        assert x.ndim == 4, "输入x不是四维的！"
        self.input = x
        N, in_chs, H, W = x.shape
        K = self.kernel_size
        pad = self.pad
        stride = self.stride
        out_chs = self.out_chs

        # pad为0相当于没有填充，大于0就填充
        self.input_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        self.raw_out = np.zeros((N, out_chs, H+2*pad-K+1, W+2*pad-K+1))

        for n in range(N):
            for k in range(out_chs):
                # out是in_chs * H' * W'维的
                out = fftconvolve(self.input_pad[n], self.weight[k])
                self.raw_out[n, k] = out.sum(axis=0)  # + self.bias[k]

        self.output = self.raw_out[:, :, ::stride, ::stride]
        return self.output

    def backward(self, delta):
        """delta和self.output形状相同，N * out_chs * H_o * W_o"""
        N, in_chs, H, W = self.input.shape
        N, out_chs, H_o, W_o = delta.shape
        K = self.kernel_size
        pad = self.pad
        stride = self.stride

        # 计算bias梯度, 没问题
        # self.bias.grad += np.sum(delta, axis=(0, 2, 3))
        # self.bias.grad /= N

        # 计算w的梯度，没问题
        raw_delta = np.zeros_like(self.raw_out)
        raw_delta[:, :, ::stride, ::stride] = delta

        for k in range(out_chs):
            for n in range(N):
                # out形状in_chs　*　K　*　K
                out = fftconvolve(self.input_pad[n], raw_delta[n, k:k+1])
                self.weight.grad[k] += out

        # 计算x的梯度
        self.grad_input = np.zeros_like(self.input, dtype=float)
        raw_delta_pad = np.pad(raw_delta, ((0, 0), (0, 0), (K-pad-1, K-pad-1), (K-pad-1, K-pad-1)))
        
        for n in range(N):
            for k in range(out_chs):
                out = fftconvolve(raw_delta_pad[n, k:k+1], self.weight[k], rotate=True)
                self.grad_input[n] += out

        return self.grad_input


    def __str__(self):
        classname = self.__class__.__name__
        return f'{classname}: in_chs={self.in_chs}, out_chs={self.out_chs}, kernel_size={self.kernel_size}, stride={self.stride}, pad={self.pad}\n'


# if __name__ == "__main__":
#     net = Conv2d(in_chs=2, out_chs=2, kernel_size=2, stride=2, pad=0)
#     input = np.arange(2*2*5*5).reshape(2, 2, 5, 5)
#     for p in net.parameters():
#         init.const_(p, 1)

#     # print(input)
#     output = net(input)
#     # print(output)

#     delta = np.random.randn(*output.shape)
#     delta = net.backward(delta)
