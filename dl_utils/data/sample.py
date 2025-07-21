# -*- coding: utf-8 -*-
# @Time    : 7/21/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : sample.py

from collections.abc import Sequence
from typing import Union, List, Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def sample_evenly(
        input_data: Union[List[Any], np.ndarray, 'torch.Tensor', Sequence[Any]],
        n: int
) -> Union[np.ndarray, List[Any], 'torch.Tensor']:
    """
    Evenly sample N elements from input_data. Supports list, numpy array, or torch tensor.

    Args:
        input_data: List, numpy array, or torch tensor to sample from.
        n: Number of elements to sample.

    Returns:
        Sampled data in the same type as input_data.
    """
    if isinstance(input_data, np.ndarray):
        length = input_data.shape[0]
        if length == 0 or n <= 0:
            return input_data[:0]
        indices = np.linspace(0, length - 1, num=n).round().astype(int)
        return input_data[indices]
    elif torch and isinstance(input_data, torch.Tensor):
        length = input_data.size(0)
        if length == 0 or n <= 0:
            return input_data[:0]
        indices = torch.linspace(0, length - 1, steps=n).round().long()
        return input_data[indices]
    else:
        # Treat as a list or other iterable data structure with length.
        length = len(input_data)
        if length == 0 or n <= 0:
            return []
        indices = np.linspace(0, length - 1, num=n).round().astype(int)
        return [input_data[i] for i in indices]


__all__ = [
    "sample_evenly"
]
