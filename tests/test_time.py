# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 19:29
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_time.py

import time

from dl_utils import Timer, get_timestamp


def test_get_timestamp():
    print(get_timestamp())


def test_timer():
    timer = Timer()

    time.sleep(0.5)
    timer("Stage 1")
    time.sleep(0.1)
    timer("Stage 2")

    timer.print()
    print(timer)
