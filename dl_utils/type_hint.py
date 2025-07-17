# -*- coding: utf-8 -*-
# @Time    : 7/17/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : type_hint.py

from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import torch

FilePath = Union[str, PathLike[str], Path]

ArrayLike = Union[np.ndarray, torch.Tensor]

__all__ = ["FilePath", "ArrayLike"]
