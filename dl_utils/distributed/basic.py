# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : basic.py

import os
from typing import Union

import torch
import torch.distributed as dist


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get (global) world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def get_master_addr() -> Union[str, None]:
    return os.environ.get("MASTER_ADDR", None)


def get_master_port() -> Union[int, None]:
    port = os.environ.get("MASTER_PORT", None)
    if port is not None:
        return int(port)
    else:
        return None


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)


def dist_breakpoint(rank: int = 0):
    """
    Breakpoint for distributed training.
    """
    assert 0 <= rank < get_world_size(), f"Invalid rank {rank}, world size: {get_world_size()}."
    if get_local_rank() == rank:
        breakpoint()
    barrier_if_distributed()


__all__ = [
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "get_device",
    "get_master_addr",
    "get_master_port",
    "barrier_if_distributed",
    "dist_breakpoint"
]
