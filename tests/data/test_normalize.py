# -*- coding: utf-8 -*-
# @Time    : 10/10/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_normalize.py


import numpy as np
import pytest
import torch

from dl_utils import normalize, inv_normalize


@pytest.mark.parametrize("dtype", [np.float32, torch.float32])
@pytest.mark.parametrize("mean,std", [
    (128, 64),  # scalar
    (np.array([0, 128, 255]), np.array([1, 64, 255])),  # vector
    (torch.tensor([0, 128, 255]), torch.tensor([1, 64, 255]))  # torch tensor vector
])
def test_normalize_inv_normalize(dtype, mean, std):
    # Create a dummy image/video: [2, 2, 3] shape
    if dtype == np.float32:
        img = np.array([[[0, 128, 255],
                         [64, 128, 192]]], dtype=np.float32)
    else:
        img = torch.tensor([[[0, 128, 255],
                             [64, 128, 192]]], dtype=torch.float32)

    # Normalize
    norm_img = normalize(img, mean, std, dim=-1)

    # Inverse normalize
    recovered_img = inv_normalize(norm_img, mean, std, dim=-1)

    # Convert both to numpy for comparison
    img_np = img.numpy() if isinstance(img, torch.Tensor) else img
    rec_np = recovered_img.numpy() if isinstance(recovered_img, torch.Tensor) else recovered_img

    # Check that recovered image matches original
    np.testing.assert_allclose(rec_np, img_np, rtol=1e-5)


@pytest.mark.parametrize("img, mean, std, dim", [
    # NumPy image, scalar mean/std, last dim
    (np.array([[[0, 128, 255]]], dtype=np.float32), 128, 64, -1),
    # NumPy image, vector mean/std, last dim
    (np.array([[[0, 128, 255]]], dtype=np.float32),
     np.array([0, 128, 255]), np.array([1, 64, 255]), -1),
    # Torch tensor, scalar mean/std, last dim
    (torch.tensor([[[0, 128, 255]]], dtype=torch.float32), 128, 64, -1),
    # Torch tensor, vector mean/std, last dim
    (torch.tensor([[[0, 128, 255]]], dtype=torch.float32),
     torch.tensor([0, 128, 255]), torch.tensor([1, 64, 255]), -1),
])
def test_normalize(img, mean, std, dim):
    result = normalize(img, mean, std, dim=dim)

    # Convert to numpy for checking
    result_np = result.numpy() if isinstance(result, torch.Tensor) else result
    mean_np = mean.numpy() if isinstance(mean, torch.Tensor) else mean
    std_np = std.numpy() if isinstance(std, torch.Tensor) else std

    # Broadcast if needed
    if np.ndim(mean_np) == 1:
        shape = [1] * result_np.ndim
        shape[dim] = -1
        mean_np = mean_np.reshape(shape)
        std_np = std_np.reshape(shape)

    expected = ((img.numpy() if isinstance(img, torch.Tensor) else img) - mean_np) / std_np
    np.testing.assert_allclose(result_np, expected, rtol=1e-5)


@pytest.mark.parametrize("img, mean, std, dim", [
    # NumPy image, scalar mean/std
    (np.array([[[0.0, 1.0, -1.0]]], dtype=np.float32), 128, 64, -1),
    # NumPy image, vector mean/std
    (np.array([[[0.0, 1.0, -1.0]]], dtype=np.float32),
     np.array([0, 128, 255]), np.array([1, 64, 255]), -1),
    # Torch tensor, scalar mean/std
    (torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.float32), 128, 64, -1),
    # Torch tensor, vector mean/std
    (torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.float32),
     torch.tensor([0, 128, 255]), torch.tensor([1, 64, 255]), -1),
])
def test_inv_normalize(img, mean, std, dim):
    result = inv_normalize(img, mean, std, dim=dim)

    # Convert to numpy for checking
    result_np = result.numpy() if isinstance(result, torch.Tensor) else result
    mean_np = mean.numpy() if isinstance(mean, torch.Tensor) else mean
    std_np = std.numpy() if isinstance(std, torch.Tensor) else std

    # Broadcast if needed
    if np.ndim(mean_np) == 1:
        shape = [1] * result_np.ndim
        shape[dim] = -1
        mean_np = mean_np.reshape(shape)
        std_np = std_np.reshape(shape)

    expected = (img.numpy() if isinstance(img, torch.Tensor) else img) * std_np + mean_np
    np.testing.assert_allclose(result_np, expected, rtol=1e-5)
