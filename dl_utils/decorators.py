# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : decorators.py

import logging
from typing import Callable

from .distributed import barrier_if_distributed

logger = logging.getLogger(__name__)


def log_on_entry(func: Callable) -> Callable:
    """
    Functions with this decorator will log the function name at entry.
    When using multiple decorators, this must be applied innermost to properly capture the name.
    """

    def log_on_entry_wrapper(*args, **kwargs):
        logger.info(f"Entering {func.__name__}")
        return func(*args, **kwargs)

    return log_on_entry_wrapper


def barrier_on_entry(func: Callable) -> Callable:
    """
    Functions with this decorator will start executing when all ranks are ready to enter.
    """

    def barrier_on_entry_wrapper(*args, **kwargs):
        barrier_if_distributed()
        return func(*args, **kwargs)

    return barrier_on_entry_wrapper


__all__ = ["log_on_entry", "barrier_on_entry"]
