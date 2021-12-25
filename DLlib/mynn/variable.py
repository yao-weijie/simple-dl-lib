import numpy as np


class Ndarry(np.ndarray):
    def __init__(self, shape: np.shape, dtype: np.dtype = np.float):
        super(Ndarry, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = False
        self.grad = np.zeros(shape, dtype)

    def zero_grad(self) -> None:
        self.grad.setfield(0, dtype=np.float)


def Param(arr: np.ndarray, requires_grad: bool = True) -> Ndarry:
    param = Ndarry(arr.shape)
    param.setfield(arr, arr.dtype)
    param.requires_grad = requires_grad
    return param
