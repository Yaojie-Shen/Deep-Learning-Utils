# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 19:29
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_timer.py

import time

from dl_utils import get_timestamp, get_readable_timestamp, ExecutionTimer


def test_get_timestamp():
    print(get_timestamp())


def test_get_readable_timestamp():
    print(get_readable_timestamp())


def test_timer():
    timer = ExecutionTimer(log=True)

    print()

    with timer.stage("All"):
        timer.start_stage("Stage 1")
        time.sleep(0.1)
        timer.end_stage("Stage 1")
        timer.start_stage()
        time.sleep(0.2)
        timer.end_stage("Stage 2")
        time.sleep(0.2)
        timer.end_stage("Stage 2")

        timer.start_stage("Stage 3")
        time.sleep(0.3)
        timer.start_stage()
        time.sleep(0.4)
        timer.end_stage("Stage 4")

    print()
    timer.print_table()
    print()
    print(timer)
