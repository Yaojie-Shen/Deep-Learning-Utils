# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:42
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_video.py

import os

import numpy as np
import torch
from pytest import mark

from dl_utils import get_duration_info, save_video


def test_save_video():
    v = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    save_video(v, "test_output/test_save_video_numpy.mp4", 30)
    v = torch.randint(0, 255, (10, 128, 128, 3), dtype=torch.uint8)
    save_video(v, "test_output/test_save_video_torch.mp4", 30)


@mark.parametrize(
    "video_root", [
        # the folder contains test videos
        ""
    ]
)
def test_get_duration_info(video_root):
    videos = [os.path.join(video_root, f) for f in os.listdir(video_root)]

    # single
    print("single:")
    print(get_duration_info(videos[0]))
    print("batch:")
    print(get_duration_info(videos[:10]))
