import numpy as np

from .variable import AGarry


def uniform_(arr: AGarry, a=0.0, b=1.0):
    tmp_arr = np.random.rand(*arr.shape) * (b - a) + a
    arr.setfield(tmp_arr, dtype=tmp_arr.dtype)


def normal_(arr: AGarry, mean=0.0, std=1.0):
    tmp_arr = np.random.randn(*arr.shape) * std + mean
    arr.setfield(tmp_arr, dtype=tmp_arr.dtype)


def const_(arr: AGarry, val):
    arr.setfield(val, dtype=np.float)
