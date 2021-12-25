import numpy as np
from .basemodule import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, label):
        assert len(x) == len(label)
        self.input = x
        label = np.reshape(label, (len(label), -1))
        self.label = label
        self.output = 0.5 * np.sum((x-label) ** 2)# / len(x)
        return self.output

    def backward(self):
        self.grad_input = (self.input - self.label)# / len(self.input)
        return self.grad_input


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, label):
        """
        x: N * C
        label: N, (1 dim array)
        """
        assert len(x) == len(label)
        self.input = x
        self.label = label
        N = len(label)
        self.q = np.exp(x) / (np.exp(x).sum(axis=1, keepdims=True) + 1e-20)  # softmax operation
        log_q = np.log(self.q)
        log_qk = log_q[list(range(N)), label]
        self.output =  -log_qk.sum()

        return self.output

    def backward(self):
        self.grad_input = np.zeros_like(self.input)
        self.grad_input += self.q
        self.grad_input[list(range(len(self.label))), self.label] -= 1

        return self.grad_input
