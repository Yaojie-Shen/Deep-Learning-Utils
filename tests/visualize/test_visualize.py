# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:00
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_visualize.py

import numpy as np

from dl_utils import plot_distribution


def test_plot_distribution():
    # generate random data
    data = np.random.default_rng(1).standard_normal(1000)

    plot_distribution(data, remove_outlier=True)
