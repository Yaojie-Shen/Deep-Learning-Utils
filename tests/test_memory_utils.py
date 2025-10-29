# -*- coding: utf-8 -*-
# @Time    : 10/29/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_memory_utils.py

import time

from dl_utils import measure_memory


def test_measure_memory():
    with measure_memory("test") as m:
        time.sleep(0.1)

    print(m.format_message())
    print(m.format_message(include_cpu=False))
    print(m.format_message(include_cpu=True))

    print(m.as_dict())


def test_measure_memory_no_tag():
    with measure_memory() as m:
        time.sleep(0.1)
    print(m.format_message())
    print(m.format_message(include_cpu=False))
    print(m.format_message(include_cpu=True))
