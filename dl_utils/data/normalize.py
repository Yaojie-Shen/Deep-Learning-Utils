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

from .. import to_numpy, to_original
from ..type_hint import ArrayLike


def _prepare_mean_std(
        mean: float | int | np.ndarray | torch.Tensor,
        std: float | int | np.ndarray | torch.Tensor,
        dim: int,
        ndim: int
) -> tuple[np.ndarray, np.ndarray]:
    """A helper function to prepare mean and std for normalization."""
    # dtype -> numpy
    if isinstance(mean, (int, float)):
        mean = np.array([mean])
    else:
        mean = to_numpy(mean)
    if isinstance(std, (int, float)):
        std = np.array([std])
    else:
        std = to_numpy(std)

    # Reshape mean and std to broadcast along the specified dim
    shape = [1] * ndim
    shape[dim] = -1
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return mean, std


def normalize(
        data: ArrayLike,
        mean: float | int | np.ndarray | torch.Tensor,
        std: float | int | np.ndarray | torch.Tensor,
        dim=-1
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

    # Convert to numpy for simplicity
    ori_type = type(data)
    data = to_numpy(data)

    mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))

    # Normalize
    data = (data - mean) / std

    # Convert back to original type
    return to_original(data, ori_type)


def inv_normalize(
        data: ArrayLike,
        mean: float | int | np.ndarray | torch.Tensor,
        std: float | int | np.ndarray | torch.Tensor,
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
    # Convert to numpy for simplicity
    ori_type = type(data)
    data = to_numpy(data)

    mean, std = _prepare_mean_std(mean, std, dim, len(data.shape))

    # Inverse normalize
    data = data * std + mean

    # Convert back to original type
    return to_original(data, ori_type)
