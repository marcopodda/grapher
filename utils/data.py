import torch
import numpy as np

from .constants import SOS, EOS


def to_sorted_tensor(lst, order):
    return [torch.LongTensor(lst[i]) for i in order]


def reverse_argsort(lst):
    arr = np.array(lst)
    return (-arr).argsort()


def pad_right(arr, pad):
    if isinstance(arr, tuple):
        return arr + (pad,)
    return arr + [pad]


def pad_left(arr, pad):
    if isinstance(arr, tuple):
        return (pad,) + arr
    return [pad] + arr
