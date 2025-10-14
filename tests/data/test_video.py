# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:42
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_video.py

import time
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from pytest import mark, fixture

from dl_utils import get_video_duration_batch, save_video, load_video


def test_save_video():
    v = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    save_video(v, "test_output/test_save_video_numpy.mp4", 30)
    v = torch.randint(0, 255, (10, 128, 128, 3), dtype=torch.uint8)
    save_video(v, "test_output/test_save_video_torch.mp4", 30)


@mark.parametrize(
    "video", [
        "test_output/test_save_video_numpy.mp4",
        "test_output/test_save_video_torch.mp4"
    ]
)
def test_get_duration_info(video):
    # single
    s_time = time.time()
    for _ in range(10):
        print(get_video_duration_batch([video]))
    print(f"Sequential took {time.time() - s_time}s | {10 / (time.time() - s_time)} s/video")

    # parallel
    s_time = time.time()
    print(get_video_duration_batch([video] * 10))
    print(f"Parallel took {time.time() - s_time}s")


@fixture
def mock_videocap():
    fake_frame = np.ones((7, 11, 3), dtype=np.uint8) * 255
    mock_instance = MagicMock()
    mock_instance.read.side_effect = [(True, fake_frame)] * 10 + [(False, None)]

    with patch("cv2.VideoCapture", return_value=mock_instance):
        yield


def test_load_video_mocked(mock_videocap):
    frames = load_video("dummy.mp4", max_frames=5)
    assert frames.shape == (5, 7, 11, 3)


def test_load_video_resize(mock_videocap):
    frames = load_video("dummy.mp4", resize=(3, 5))  # W, H
    assert frames.shape == (10, 5, 3, 3)  # F, H, W, C


def test_load_video_resize_int(mock_videocap):
    frames = load_video("dummy.mp4", resize=14)
    assert frames.shape == (10, 14, 22, 3)


def test_load_video_center_crop(mock_videocap):
    frames = load_video("dummy.mp4", center_crop=(3, 5))
    assert frames.shape == (10, 5, 3, 3)


def test_load_video_center_crop_int(mock_videocap):
    frames = load_video("dummy.mp4", center_crop=2)
    assert frames.shape == (10, 2, 2, 3)
