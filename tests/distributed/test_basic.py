# -*- coding: utf-8 -*-
# @Time    : 10/10/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_basic.py

from dataclasses import dataclass

import pytest

from dl_utils import dist_info


@dataclass
class MockDist:
    _is_initialized: bool

    def is_initialized(self): return self._is_initialized

    def get_backend(self): return "nccl"

    def get_world_size(self): return 8

    def get_rank(self): return 2

    def get_master_addr(self): return "127.0.0.1"

    def get_master_port(self): return 12345


@pytest.fixture
def mock_dist(monkeypatch):
    """Mock torch.distributed functions."""

    mock = MockDist(True)
    monkeypatch.setattr("torch.distributed.is_initialized", mock.is_initialized)
    monkeypatch.setattr("torch.distributed.get_backend", mock.get_backend)
    monkeypatch.setattr("torch.distributed.get_world_size", mock.get_world_size)
    monkeypatch.setattr("torch.distributed.get_rank", mock.get_rank)
    monkeypatch.setattr("dl_utils.distributed.basic.get_master_addr", mock.get_master_addr)
    monkeypatch.setattr("dl_utils.distributed.basic.get_master_port", mock.get_master_port)
    return mock


@pytest.fixture
def mock_dist_not_initialized(monkeypatch):
    """Mock torch.distributed functions when NOT initialized."""

    mock = MockDist(False)
    monkeypatch.setattr("torch.distributed.is_initialized", mock.is_initialized)
    return mock


def test_print_dist_info(monkeypatch, mock_dist):
    print()
    dist_info()
    dist_info(prefix="Worker")


def test_print_dist_info_not_initialized(monkeypatch, mock_dist_not_initialized):
    print()
    dist_info()
    dist_info(prefix="Worker")
