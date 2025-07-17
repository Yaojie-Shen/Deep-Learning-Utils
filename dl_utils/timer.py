# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 13:41
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : timer.py

import datetime
import time
from collections import defaultdict
from collections import deque
from contextlib import contextmanager
from typing import Optional

from tabulate import tabulate


def get_timestamp():
    return "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.datetime.now())


def get_readable_timestamp():
    return "{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())


def get_current_time_in_ms(precision: int) -> float:
    """Return the current time in milliseconds."""
    return round(time.time() * 1000, precision)


class Timer:
    def __init__(self, precision=3):
        self._precision = precision
        self._start_time = self.get_current_time_in_ms()

    def get_current_time_in_ms(self) -> float:
        """Return the current time in milliseconds."""
        return get_current_time_in_ms(self._precision)

    def reset(self) -> float:
        self._start_time = self.get_current_time_in_ms()
        return self._start_time

    def get_duration(self) -> float:
        return self.get_current_time_in_ms() - self._start_time

    def get_duration_and_reset(self) -> float:
        duration = self.get_duration()
        self.reset()
        return duration


class ExecutionTimer(Timer):

    def __init__(
            self, history_size=None, precision=2,
            start_prompt: str = None, end_prompt: str = None, log: bool = False
    ):
        super().__init__(precision=precision)

        self._stage_name = None
        self._stage_log = defaultdict(lambda: deque(maxlen=history_size))

        self._enable_log = log
        self._start_log_prompt = \
            "({ctime}) => Starting stage: {stage}..." if start_prompt is None else start_prompt
        self._end_log_prompt = \
            "({ctime}) => Finished stage: {stage} | Took {duration:.3f} ms." if end_prompt is None else end_prompt

    @contextmanager
    def stage(self, name: str):
        """
        Usage:
            with timer.stage("stage_name"):
                # do something

        Note: This context manager stores the stage name and duration independently, so it is suitable for nested use.
        """
        start = self.get_current_time_in_ms()
        try:
            yield start
        finally:
            duration = self.get_current_time_in_ms() - start
            self._stage_log[name].append(duration)

    def start_stage(self, name: Optional[str] = None):
        """
        Log the start of a stage.
        Note: Only for sequential use.

        Args:
            name: The name of the stage to start. If None, one must be specified in the next call to `end_stage`.
                If not None and there is a previous stage, the previous stage will be ended automatically.
        """
        assert name is None or isinstance(name, str), \
            f"Invalid stage name: {name}"

        # Previous stage is not finished
        if self._stage_name is not None:
            self.end_stage()  # End previous stage

        # Set current stage name
        self._stage_name = name
        self.reset()

        if self._enable_log and name is not None:
            print(self._start_log_prompt.format(ctime=get_readable_timestamp(), stage=self._stage_name))

    def end_stage(self, name: Optional[str] = None):
        """
        Log the end of a stage.
        Note: Only for sequential use.

        Args:
            name: The name of the stage to end. If None, the name of the last call to `start_stage` must be specified.
        """
        assert name is None or isinstance(name, str), \
            f"Invalid stage name: {name}"
        assert name is not None or self._stage_name is not None, \
            "Stage name is unknown: it must be set once at the beginning or end of the stage"

        if name is None:
            name = self._stage_name
        elif self._stage_name is not None:
            assert self._stage_name == name, \
                f"Stage name mismatch: {self._stage_name} (in process) != {name} (trying to end)"

        duration = self.get_duration_and_reset()
        self._stage_log[name].append(duration)
        self._stage_name = None

        if self._enable_log:
            print(self._end_log_prompt.format(ctime=get_readable_timestamp(), stage=name, duration=duration))

    def __str__(self):
        out = ""
        info = self.summary()
        for stage_name, stage_info in info.items():
            out += f"[{stage_name}]:\n\ttotal {stage_info['total']} ms\n" \
                   f"\t{stage_info['count']}-iters range ({stage_info['min']}, {stage_info['max']})\n" \
                   f"\tavg {stage_info['avg']} ms\n"
        return out

    def summary(self):
        return {
            k: {
                "total": round(sum(v), self._precision),
                "avg": round(sum(v) / len(v), self._precision),
                "min": round(min(v), self._precision),
                "max": round(max(v), self._precision),
                "last": round(v[-1], self._precision),
                "count": len(v),
            } for k, v in self._stage_log.items()
        }

    def print_table(self):
        data = [[k, v["total"], v["count"], v["min"], v["max"], v["avg"]]
                for k, v in self.summary().items()]
        print(tabulate(
            data,
            headers=["Stage", "Total (ms)", "Count", "Min (ms)", "Max (ms)", "Avg (ms)"],
            tablefmt="simple"
        ))


__all__ = ["get_timestamp", "get_readable_timestamp", "Timer", "ExecutionTimer"]
