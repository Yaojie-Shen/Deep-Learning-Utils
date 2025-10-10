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

ArrayLike = Union["np.ndarray", "torch.Tensor"]

__all__ = ["FilePath", "ArrayLike"]
