# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:42
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_video.py

import os

from pytest import mark

from dl_utils import get_duration_info


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
