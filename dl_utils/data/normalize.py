# -*- coding: utf-8 -*-
# @Time    : 9/8/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : normalize.py

__all__ = [
    "normalize",
    "inv_normalize"
]

import numpy as np
import torch

from .. import to_numpy, to_tensor
from ..type_hint import ArrayLike, Scalar, TorchOrNumpy


def _prepare_mean_std(
        mean: TorchOrNumpy,
        std: TorchOrNumpy,
        dim: int,
        ndim: int
) -> tuple[TorchOrNumpy, TorchOrNumpy]:
    """A helper function to prepare mean and std for normalization.
    Reshape mean and std to broadcast along the specified dim
    """
    shape = [1] * ndim
    shape[dim] = -1
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return mean, std


def _normalize(data, mean, std):
    return (data - mean) / std


def _inv_normalize(data, mean, std):
    return data * std + mean


def normalize(
        data: ArrayLike,
        mean: Scalar | ArrayLike,
        std: Scalar | ArrayLike,
        dim: int = -1
) -> ArrayLike:
    """Normalize the input array (usually image or video).

    Args:
        data: Input array, can be a NumPy array or a PyTorch tensor.
        mean: Scalar or vector of means for each channel.
        std: Scalar or vector of standard deviations for each channel.
        dim: The channel dimension to normalize along. Default is -1 (last dimension).

    Returns:
        Normalized image or video in the same type as input (NumPy array or PyTorch tensor).

    Examples:
        >>> import numpy as np
        >>> from dl_utils import normalize
        >>> img = np.array([[[0, 128, 255]]], dtype=np.float32)
        >>> normalize(img, mean=128, std=64)
        array([[[-2.      ,  0.      ,  1.984375]]])

        >>> import torch
        >>> from dl_utils import normalize
        >>> img_t = torch.tensor([[[0, 128, 255]]], dtype=torch.float32)
        >>> normalize(img_t, mean=torch.tensor([0, 128, 255]), std=torch.tensor([1, 64, 255]))
        tensor([[[0., 0., 0.]]], dtype=torch.float64)
    """

    if isinstance(data, list):
        return normalize(np.array(data), mean, std, dim).tolist()
    if isinstance(data, tuple):
        return tuple(normalize(np.array(data), mean, std, dim).tolist())
    elif isinstance(data, np.ndarray):
        mean, std = to_numpy(mean), to_numpy(std)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _normalize(data, mean, std)
    elif isinstance(data, torch.Tensor):
        # NOTE: Move std and mean to same device and convert to same dtype
        mean, std = to_tensor(mean).to(data), to_tensor(std).to(data)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _normalize(data, mean, std)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def inv_normalize(
        data: ArrayLike,
        mean: Scalar | ArrayLike,
        std: Scalar | ArrayLike,
        dim=-1
) -> ArrayLike:
    """Inverse normalize the input array (usually image or video).

    Args:
        data: Input array, can be a NumPy array or a PyTorch tensor,
              which has been previously normalized.
        mean: Scalar or vector of means used in the original normalization.
        std: Scalar or vector of standard deviations used in the original normalization.
        dim: The channel dimension along which normalization was applied.
             Default is -1 (last dimension).

    Returns:
        Denormalized image or video in the same type as input (NumPy array or PyTorch tensor).
    """
    if isinstance(data, list):
        return inv_normalize(np.array(data), mean, std, dim).tolist()
    if isinstance(data, tuple):
        return tuple(inv_normalize(np.array(data), mean, std, dim).tolist())
    elif isinstance(data, np.ndarray):
        mean, std = to_numpy(mean), to_numpy(std)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _inv_normalize(data, mean, std)
    elif isinstance(data, torch.Tensor):
        # NOTE: Move std and mean to same device and convert to same dtype
        mean, std = to_tensor(mean, to=data), to_tensor(std, to=data)
        mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))
        return _inv_normalize(data, mean, std)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
