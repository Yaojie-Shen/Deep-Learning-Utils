# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : type_hint.py

from os import PathLike
from pathlib import Path
from typing import Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

FilePath = Union[str, PathLike[str], Path]

ArrayLike = Union["np.ndarray", "torch.Tensor", list, tuple]

TorchOrNumpy = Union["np.ndarray", "torch.Tensor"]

Scalar = Union[int, float]

__all__ = ["FilePath", "ArrayLike", "TorchOrNumpy", "Scalar"]
