# -*- coding: utf-8 -*-
# @Time    : 9/12/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_global_cache_utils.py

import gc
import multiprocessing
import random
import time

from dl_utils import GlobalCache


def test_global_cache():
    cache = GlobalCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)

    cache2 = GlobalCache()
    assert cache2.get("a") == 1
    assert cache2.get("b") == 2
    assert cache2.get("c") is None

    del cache, cache2
    gc.collect()

    cache3 = GlobalCache()
    assert cache3.get("a") == 1
    assert cache3.get("b") == 2
    assert cache3.get("c") is None


def worker(key, value, return_dict):
    cache = GlobalCache()
    time.sleep(random.random() * 0.1)
    cache.set(key, value)
    # Return the value from this process
    return_dict[key] = cache.get(key)


def test_global_cache_multiprocessing():
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(100):
        p = multiprocessing.Process(target=worker, args=(f"key{i}", i, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Ensure values set in one process are not seen by other processes
    # The return_dict contains values from each process only
    for i in range(100):
        assert return_dict[f"key{i}"] == i
