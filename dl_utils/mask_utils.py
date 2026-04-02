# -*- coding: utf-8 -*-
# @Time    : 3/20/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : mask_utils.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_THRESHOLD = 127


def binarize_mask(mask, threshold: int = DEFAULT_THRESHOLD):
    """Convert a mask to a 2D boolean array.

    Accepts:
    - (H, W) bool / numeric
    - (H, W, 1)
    - (H, W, 3/4) (treated as any-channel > threshold)
    """
    # Accept numpy arrays, PIL images, etc.
    arr = np.asarray(mask)

    if arr.dtype == np.bool_:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        raise ValueError(f"Unsupported mask shape for bool mask: {arr.shape}")

    if arr.ndim == 2:
        return arr > threshold

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0] > threshold
        if arr.shape[-1] in (3, 4):
            return (arr > threshold).any(axis=-1)

    raise ValueError(f"Unsupported mask shape: {arr.shape}")


def unbinarize_mask(mask, true_value: int = 255, false_value: int = 0, dtype=np.uint8):
    """Convert a 2D boolean mask back to a uint8 (or other dtype) mask.

    This is the reverse of :func:`binarize_mask` when the input mask is already boolean.
    The output will be a 2D array with values in {false_value, true_value}.
    """
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        raise ValueError(f"`unbinarize_mask` expects a boolean mask, got dtype={arr.dtype} shape={arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"`unbinarize_mask` expects a 2D mask, got shape={arr.shape}")

    out = np.full(arr.shape, false_value, dtype=dtype)
    out[arr] = true_value
    return out


def load_mask(
        path: str | Path,
        threshold: int = DEFAULT_THRESHOLD,
        invert: bool = False,
        as_bool: bool = True,
):
    """Read a mask image from disk.

    The input is usually a grayscale / binary image. This function converts it into:
    - bool mask if `as_bool=True`
    - uint8 mask in {0, 255} if `as_bool=False`

    """
    p = Path(path)
    img = Image.open(p)
    # Convert to grayscale to make the behavior consistent for RGB/RGBA inputs.
    img = img.convert("L")
    arr = np.asarray(img)

    m = arr > threshold
    if invert:
        m = invert_mask(m)

    if as_bool:
        return m
    return m.astype(np.uint8) * 255


def save_mask(
        mask,
        path: str | Path,
        threshold: int = DEFAULT_THRESHOLD,
        invert: bool = False,
):
    """Save a mask to an image file.

    The output image is an 8-bit grayscale image with values in {0, 255}.
    """
    m = binarize_mask(mask, threshold=threshold)
    if invert:
        m = invert_mask(m)

    out = unbinarize_mask(m)
    # For uint8 2D arrays, Pillow will infer "L" mode.
    img = Image.fromarray(out)
    img.save(str(path))


def union_masks(
        *masks,
        threshold: int = DEFAULT_THRESHOLD,
):
    """Union (logical OR) of multiple masks."""
    if len(masks) == 0:
        raise ValueError("At least one mask is required")

    ms = [binarize_mask(m, threshold=threshold) for m in masks]
    shape = ms[0].shape
    if any(x.shape != shape for x in ms[1:]):
        raise ValueError(f"All masks must have the same shape, got: {[x.shape for x in ms]}")
    return np.logical_or.reduce(ms)


def intersect_masks(
        *masks,
        threshold: int = DEFAULT_THRESHOLD,
):
    """Intersection (logical AND) of multiple masks."""
    if len(masks) == 0:
        raise ValueError("At least one mask is required")

    ms = [binarize_mask(m, threshold=threshold) for m in masks]
    shape = ms[0].shape
    if any(x.shape != shape for x in ms[1:]):
        raise ValueError(f"All masks must have the same shape, got: {[x.shape for x in ms]}")
    return np.logical_and.reduce(ms)


def subtract_mask(
        a,
        b,
        threshold: int = DEFAULT_THRESHOLD,
):
    """Set difference: keep pixels in `a` but not in `b` (a AND (NOT b))."""
    a2 = binarize_mask(a, threshold=threshold)
    b2 = binarize_mask(b, threshold=threshold)
    if a2.shape != b2.shape:
        raise ValueError(f"Masks must have the same shape, got: {a2.shape} vs {b2.shape}")
    return a2 & (~b2)


def invert_mask(
        mask,
        threshold: int = DEFAULT_THRESHOLD,
):
    """Invert a mask (logical NOT)."""
    m2 = binarize_mask(mask, threshold=threshold)
    return ~m2


def mask_iou(a, b, threshold: int = DEFAULT_THRESHOLD) -> float:
    """Compute IoU (Intersection over Union) between two masks.

    Inputs can be numpy bool masks, numpy uint8 masks (0-255), or PIL images.
    The function always binarizes inputs first and returns a float in [0, 1].

    By convention:
    - If both masks are empty (union == 0), IoU is 1.0.
    - If union == 0 but inputs are not both empty (shouldn't happen after binarization), returns 0.0.
    """
    a2 = binarize_mask(a, threshold=threshold)
    b2 = binarize_mask(b, threshold=threshold)
    if a2.shape != b2.shape:
        raise ValueError(f"Masks must have the same shape, got: {a2.shape} vs {b2.shape}")

    inter = np.logical_and(a2, b2).sum(dtype=np.int64)
    union = np.logical_or(a2, b2).sum(dtype=np.int64)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


__all__ = [
    "binarize_mask",
    "load_mask",
    "save_mask",
    "union_masks",
    "intersect_masks",
    "subtract_mask",
    "invert_mask",
    "mask_iou",
    "unbinarize_mask",
]
