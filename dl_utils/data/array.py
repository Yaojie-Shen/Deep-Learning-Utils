# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : array.py

import numpy as np
import torch

from ..type_hint import ArrayLike


def to_numpy(array: ArrayLike) -> np.ndarray:
    """
    Convert array-like object to numpy array.
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported type: {type(array)}")


def to_tensor(array: ArrayLike) -> torch.Tensor:
    """
    Convert array-like object to torch tensor.
    """
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise TypeError(f"Unsupported type: {type(array)}")


def to_original(array: ArrayLike, ori_dtype) -> ArrayLike:
    """
    Convert array-like object to original type.
    """
    if ori_dtype is np.ndarray:
        return to_numpy(array)
    elif ori_dtype is torch.Tensor:
        return to_tensor(array)
    else:
        raise TypeError(f"Unsupported type: {ori_dtype}")


__all__ = [
    "to_numpy",
    "to_tensor",
    "to_original",
]
