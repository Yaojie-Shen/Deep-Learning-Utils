# -*- coding: utf-8 -*-
# @Time    : 7/21/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_sample.py

import numpy as np
import pytest
import torch

from dl_utils import sample_evenly


@pytest.mark.parametrize("input_data, N, expected_type", [
    ([1, 2, 3, 4, 5], 3, list),
    (np.array([1, 2, 3, 4, 5]), 3, np.ndarray),
    (torch.tensor([1, 2, 3, 4, 5]), 3, torch.Tensor),
])
def test_sample_evenly_basic(input_data, N, expected_type):
    result = sample_evenly(input_data, N)
    print(result)
    assert isinstance(result, expected_type)
    assert len(result) == 3
    # Check for even distribution
    if isinstance(result, list):
        assert result[0] == input_data[0]
        assert result[-1] == input_data[-1]
    elif isinstance(result, (np.ndarray, torch.Tensor)):
        assert result[0].item() == input_data[0].item()
        assert result[-1].item() == input_data[-1].item()


def test_sample_evenly_edge_cases():
    assert sample_evenly([], 3) == []
    assert sample_evenly(np.array([]), 3).size == 0
    assert sample_evenly(torch.tensor([]), 3).numel() == 0

    assert sample_evenly([1, 2], 0) == []
    assert sample_evenly(np.array([1, 2]), 0).size == 0
    assert sample_evenly(torch.tensor([1, 2]), 0).numel() == 0


def test_sample_evenly_more_N_than_input():
    data = [10, 20]
    result = sample_evenly(data, 5)
    print(result)
    assert len(result) == 5


if __name__ == '__main__':
    pytest.main()
