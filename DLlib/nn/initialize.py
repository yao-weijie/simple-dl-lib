import numpy as np

from .variable import AGarray


def uniform_(arr: AGarray, a=0.0, b=1.0):
    tmp_arr = np.random.rand(*arr.shape) * (b - a) + a
    arr.setfield(tmp_arr, dtype=tmp_arr.dtype)


def normal_(arr: AGarray, mean=0.0, std=1.0):
    tmp_arr = np.random.randn(*arr.shape) * std + mean
    arr.setfield(tmp_arr, dtype=tmp_arr.dtype)


def const_(arr: AGarray, val):
    arr.setfield(val, dtype=np.float)
