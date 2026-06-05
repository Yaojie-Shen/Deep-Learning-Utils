# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_ray_inference.py

import time
from collections import Counter

import pytest

ray = pytest.importorskip("ray")


from dl_utils import RayActorScheduler, RayActorSchedulerWrapper  # noqa: E402


@ray.remote
class Worker:
    def __init__(self, idx: int, sleep_time: float):
        self.idx = idx
        self.sleep_time = sleep_time

    def process(self, x):
        time.sleep(self.sleep_time)
        return x, self.idx


def test_scheduler_balances_load():
    """Create 10 actors (9x 0.1s, 1x 1.5s), run 101 tasks and ensure elapsed time is ~1.5s.

    With queue_max_size=1 there are 10 concurrent slots. The 10 fast actors
    will each process ~10 tasks in ~1.0s (0.1s per task), and the slow actor will
    process 1 task in ~1.5s, so 101 tasks should finish close to 1.5 seconds.
    """
    # Start Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create 10 fast actors and 1 slow actor, wait for them to be ready before continuing
        actors = [Worker.remote(idx, 0.1) for idx in range(10)]
        actors.append(Worker.remote(10, 1.5))

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
        assert schdule_count[10] == 1, (
            f"Slow worker should process 1 task, but processed {schdule_count[10]}"
        )
    finally:
        ray.shutdown()


def test_scheduler_wrapper_supports_actor_like_remote_calls():
    ray.init(ignore_reinit_error=True)

    try:
        actors = [Worker.remote(idx, 0.0) for idx in range(2)]
        wrapper = RayActorSchedulerWrapper(actors, queue_max_size=1)

        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(wrapper._scheduler._scheduler_actor.__ray_ready__.remote())

        refs = [wrapper.process.remote(i) for i in range(8)]
        results = ray.get(refs)

        assert [x for x, _ in results] == list(range(8))
        assert {worker_idx for _, worker_idx in results}.issubset({0, 1})
    finally:
        ray.shutdown()


def test_scheduler_wrapper_supports_direct_call_alias():
    ray.init(ignore_reinit_error=True)

    try:
        actors = [Worker.remote(idx, 0.0) for idx in range(2)]
        wrapper = RayActorSchedulerWrapper(actors, queue_max_size=1)

        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(wrapper._scheduler._scheduler_actor.__ray_ready__.remote())

        refs = [wrapper.process(i) for i in range(8)]
        results = ray.get(refs)

        assert [x for x, _ in results] == list(range(8))
        assert {worker_idx for _, worker_idx in results}.issubset({0, 1})
    finally:
        ray.shutdown()


def test_scheduler_wrapper_resolves_missing_methods_lazily():
    ray.init(ignore_reinit_error=True)

    try:
        actors = [Worker.remote(0, 0.0)]
        wrapper = RayActorSchedulerWrapper(actors, queue_max_size=1)

        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(wrapper._scheduler._scheduler_actor.__ray_ready__.remote())

        ref = wrapper.missing_method.remote(1)

        assert isinstance(ref, ray.ObjectRef)
        with pytest.raises(Exception):
            ray.get(ref)
    finally:
        ray.shutdown()
