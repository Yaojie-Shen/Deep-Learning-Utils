# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_ray_inference.py

import time
from collections import Counter

import pytest

ray = pytest.importorskip("ray")

from dl_utils.inference.ray_inference_utils import RayActorScheduler


@ray.remote
class Worker:
    def __init__(self, idx: int, sleep_time: float):
        self.idx = idx
        self.sleep_time = sleep_time

    def process(self, x):
        time.sleep(self.sleep_time)
        return x, self.idx


def test_scheduler_balances_load():
    """Create 10 actors (9x 0.1s, 1x 1.0s), run 91 tasks and ensure elapsed time is ~1s.

    With queue_max_size=1 there are 10 concurrent slots. The 9 fast actors
    will each process ~10 tasks in ~1s (0.1s per task), and the slow actor will
    process 1 task in ~1s, so 91 tasks should finish close to 1 second.
    """
    # Start Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create 9 fast actors and 1 slow actor, wait for them to be ready before continuing
        actors = [Worker.remote(idx, 0.01) for idx in range(9)]
        actors.append(Worker.remote(9, 1.0))

        # actor_fn: call the remote process method
        def actor_fn(actor, x):
            return actor.process.remote(x)

        scheduler = RayActorScheduler(actors, actor_fn, queue_max_size=1)
        refs = []

        # wait for all actors to be ready before testing submit time
        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(scheduler._scheduler_actor.__ray_ready__.remote())

        # submit task
        start_submit = time.time()
        for i in range(101):
            refs.append(scheduler.submit(i))
        submit_time = time.time() - start_submit

        # wait for all tasks to finish
        start_wait = time.time()
        schdule_count = Counter()
        for i, ref in enumerate(refs):
            x, worker_idx = ray.get(ref)
            assert x == i, f"Expected {i} but got {x}"
            schdule_count[worker_idx] += 1
        wait_time = time.time() - start_wait

        print(f"Submit time: {submit_time:.2f}s, Wait time: {wait_time:.2f}s")
        print(f"Schedule count per worker: {schdule_count}")
        assert submit_time < 0.5, f"Submit should be fast, but took {submit_time:.2f}s"
        assert wait_time < 3, f"Tasks should finish in ~s, but took {wait_time:.2f}s"
        assert schdule_count[9] == 1, (
            f"Slow worker should process 1 task, but processed {schdule_count[9]}"
        )
    finally:
        ray.shutdown()
