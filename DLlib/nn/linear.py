import numpy as np

from .basemodule import Module
from .variable import Param


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        self.input = x
        self.output = np.reshape(x, (len(x), -1))

        return self.output

    def backward(self, delta):
        self.grad_input = np.reshape(delta, self.input.shape)

        return self.grad_input


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.bias_ = bias
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Param(np.random.randn(in_features, out_features) * 0.01)
        self._parameters["weight"] = self.weight
        if bias:
            self.bias = Param(np.random.randn(1, out_features) * 0.01)
            self._parameters["bias"] = self.bias

    def forward(self, x):
        self.input = x
        if self.bias is not None:
            self.output = np.matmul(x, self.weight) + self.bias
        else:
            self.output = np.matmul(x, self.weight)

        return self.output

    def backward(self, delta):
        self.weight.grad += np.matmul(self.input.T, delta)
        if self.bias is not None:
            self.bias.grad += np.mean(delta, axis=0, keepdims=True)
        self.grad_input = np.matmul(delta, self.weight.T)

        return self.grad_input

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}: in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_}\n"
