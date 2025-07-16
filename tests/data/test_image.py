# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 22:41
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_image.py

import numpy as np

from dl_utils import byte_imread, byte_imwrite, visualize_image


def test_byte_imread_imwrite():
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    data = byte_imwrite(image)
    image_recover = byte_imread(data)
    assert np.equal(image, image_recover).all()
