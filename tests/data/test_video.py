# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:42
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_video.py

import time

import numpy as np
import torch
from pytest import mark

from dl_utils import get_video_duration_batch, save_video


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
