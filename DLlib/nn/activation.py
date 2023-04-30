import numpy as np

from .basemodule import Module


class Sigmod(Module):
    """Sigmod激活函数"""

    def __init__(self):
        super(Sigmod, self).__init__()

    def forward(self, x):
        self.input = x
        self.output = 1.0 / (1.0 + np.exp(-x)) + 1e-10

        return self.output

    def backward(self, grad_n):
        grad_fn = self.output * (1 - self.output)
        self.grad_input = grad_n * grad_fn

        return self.grad_input


class ReLU(Module):
    """ReLU激活函数"""

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.input = x
        self.output = np.where(x > 0, x, 0)

        return self.output

    def backward(self, grad_n):
        grad_fn = np.where(self.input > 0, 1, 0)
        self.grad_input = grad_n * grad_fn

        return self.grad_input


class LeakyReLU(Module):
    """LeakyReLU激活函数"""

    def __init__(self, alpha=0.1):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * x)

        return self.output

    def backward(self, grad_n):
        grad_fn = np.where(self.input > 0, 1, self.alpha)
        self.grad_input = grad_n * grad_fn

        return self.grad_input

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}: alpha={self.alpha}\n"
