import numpy as np
from .basemodule import Module


__all__ = ['MaxPool2d', 'MeanPool2d', 'AvgPool2d']


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, pad: int = 0):
        super(MaxPool2d, self).__init__()
        assert kernel_size - pad -1 >= 0, 'padding太大了'
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # x = self.insert_dim(x)
        # pad为0相当于没有填充，不为0就填充
        pad = self.pad
        self.input_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        # TODO: 

    def backward(self, delta):
        raise NotImplementedError


class MeanPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, pad: int = 0):
        super(MeanPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        self.input = x
        N, chs, H, W = x.shape
        pad = self.pad
        K = self.kernel_size
        stride = self.stride
        self.input_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        H_o = H + 2*pad - K + 1
        W_o = W + 2*pad - K + 1
        self.raw_output = np.zeros((N, chs, H_o, W_o))

        for h in range(H_o):
            for w in range(W_o):
                self.raw_output[:, :, h, w] = np.mean(self.input_pad[:, :, h:h+K, w:w+K], axis=(2, 3))
        self.ouput = self.raw_output[:, :, ::stride, ::stride]
        return self.ouput

    def backward(self, delta):
        N, chs, H, W = self.input.shape
        N, chs, H_o, W_o = delta.shape
        K = self.kernel_size
        pad = self.pad
        stride = self.stride
        grad_input_pad = np.zeros_like(self.input_pad)
        delta /= (K * K)
        for h in range(H_o):
            for w in range(W_o):
                grad_input_pad[:, :, h:h+K, w:w+K] += delta[:, :, h:h+1, w:w+1]
        self.grad_input = grad_input_pad[:, :, pad:pad+H, pad:pad+W]
        
        return self.grad_input


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
        delta /= (H * W)
        delta = delta.reshape(N, C, 1, 1)
        self.grad_input = np.ones_like(self.input) * delta
        
        return self.grad_input
