from typing import Sequence

import numpy as np


class AGarray(np.ndarray):
    def __init__(self, shape: Sequence[int], dtype: np.dtype = float):
        super(AGarray, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = False
        self.grad = np.zeros(shape, dtype)

    def zero_grad(self) -> None:
        self.grad.setfield(0, dtype=float)


def Param(arr: np.ndarray, requires_grad: bool = True) -> AGarray:
    param = AGarray(arr.shape)
    param.setfield(arr, arr.dtype)
    param.requires_grad = requires_grad
    return param
