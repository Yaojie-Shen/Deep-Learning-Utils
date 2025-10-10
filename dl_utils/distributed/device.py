# -*- coding: utf-8 -*-
# @Time    : 9/22/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : device.py

__all__ = [
    "recursive_to",
]

from typing import Any, Union

import torch


def recursive_to(obj: Any, device: Union[str, torch.device] = None) -> Any:
    """
    Recursively move all torch.Tensor in obj to the given device.
    Supports: Tensor, list, tuple, dict, set. Leaves other objects intact.

    Args:
        obj: The object to move.
        device: The device to move to. If None, uses the current device if gpu is available, else "cpu".

    Returns:
        The object with all torch.Tensor moved to the given device.
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(v, device) for v in obj)
    elif isinstance(obj, set):
        # sets are unordered; converting back to set might lose type/sort
        return {recursive_to(v, device) for v in obj}
    else:
        return obj
