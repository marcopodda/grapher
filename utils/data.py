import torch
import numpy as np


PAD = 0
SOS = 1
EOS = 2


def to_sorted_tensor(lst, order):
    return [torch.LongTensor(lst[i]) for i in order]


def reverse_argsort(lst):
    arr = np.array(lst)
    return (-arr).argsort()


def pad_right(arr, pad):
    return arr + (pad,)


def pad_left(arr, pad):
    return (pad,) + arr
