# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : array.py

import numpy as np
import torch

from ..type_hint import ArrayLike, Scalar


def to_numpy(array: ArrayLike | Scalar) -> np.ndarray:
    """
    Convert array-like object or scalar to NumPy array.

    Args:
        array: Array-like object or scalar to be converted.
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, (list, tuple)):
        return np.array(array)
    elif isinstance(array, (int, float)):
        return np.array(array)
    else:
        raise TypeError(f"Unsupported type: {type(array)}")


def to_tensor(array: ArrayLike | Scalar, to=None) -> torch.Tensor:
    """
    Convert a scalar or array-like object to a PyTorch tensor.

    Args:
        array: Scalar or array-like object to convert.
        to: Optional. Device, dtype, or a target tensor.
            If a tensor is provided, the resulting tensor will have the same device and dtype, following PyTorch's
            `.to()` behavior.
    """
    if isinstance(array, torch.Tensor):
        pass
    elif isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, (list, tuple)):
        array = torch.tensor(array)
    elif isinstance(array, (int, float)):
        array = torch.tensor(array)
    else:
        raise TypeError(f"Unsupported type: {type(array)}")

    if to is not None:
        array = array.to(to)
    return array


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
