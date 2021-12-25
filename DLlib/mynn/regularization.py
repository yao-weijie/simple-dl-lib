import numpy as np
from .basemodule import Module
from .variable import Param


class Dropout(Module):
    def __init__(self, p: float = 0.5,
                       inplace: bool = False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.mask = None

    def forward(self,  x):
        self.input = x
        if self.training:
            shape = self.input.shape
            self.mask = np.random.binomial(n=1, p=1-self.p, size=shape)
            self.output = self.mask * self.input / (1 - self.p)
            if self.inplace:
                self.input = self.mask * self.input
        else:
            self.output = self.input

        return self.output

    def backward(self, delta):
        self.grad_input = delta * self.mask
        return self.grad_input
    
    def __str__(self):
        classname = self.__class__.__name__
        return f'{classname}: p={self.p}\n'


class BatchNorm2d(Module):
    def __init__(self):
        super(BatchNorm2d, self).__init__()
        self.mean = None
        self.var = None
        # TODO: 维数待定
        self.gamma = Param(np.random.randn(1)*0.01, requires_grad=True)
        self.beta = Param(np.random.randn(1)*0.01, requires_grad=True)
        self.delta = 1e-6
        self._parameters['gamma'] = self.gamma
        self._parameters['beta'] = self.beta

    def forward(self, x):
        if self.training and (len(x) != 1):
            self.mean = np.mean(x, axis=0)
            self.std = np.sqrt(np.var(x, axis=0) + self.delta)
            normalized = (x - self.mean) / self.std
        else:
            self.output = x

        return self.output

    def backward(self, delta):
        if self.training and (len(self.input) != 1):
            self.grad_input = delta
        else:
            self.grad_input = delta
        
        return self.grad_input

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


if __name__ == "__main__":
    a = np.ones((100, 100))
    s = np.zeros((1000,))
    net = Dropout(p=0.1)
    for i in range(1000):
        s[i] = net(a).sum()
    print(s.mean() / a.sum())