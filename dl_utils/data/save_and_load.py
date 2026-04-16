# -*- coding: utf-8 -*-
# @Time    : 1/19/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : save_and_load.py

import json
import pickle
from pathlib import Path
from typing import Optional


def save_text(text: str, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(text)


def load_text(file) -> str:
    with open(file, "r") as fp:
        return fp.read()


class _JsonBytesEncoder(json.JSONEncoder):
    """Object of type bytes is not JSON serializable, convert it to string before saving"""

    def default(self, obj):
        if isinstance(obj, bytes):  # bytes->str
            return str(obj, encoding="utf-8")
        return json.JSONEncoder.default(self, obj)


def save_json(data, file, save_pretty=False, **kwargs):
    """Save json to file.

    - If `save_pretty=True`, write a human-readable JSON string using `json.dumps` with `indent=4` and `sort_keys=True`.
    - Extra keyword arguments forwarded to `json.dump`.
    """

    _kwargs = {"cls": _JsonBytesEncoder}
    if save_pretty:
        _kwargs.update({"indent": 4, "ensure_ascii": False})
    _kwargs.update(kwargs)

    # Dump to string first to avoid writing if error occurs
    s = json.dumps(data, **_kwargs)
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(s)


def load_json(file):
    with open(file, "r") as fp:
        return json.load(fp)


def save_jsonl(data, file, **kwargs):
    _kwargs = {"cls": _JsonBytesEncoder}
    _kwargs.update(kwargs)
    lines = []
    for idx, item in enumerate(data):
        line = json.dumps(item, **_kwargs)
        if "\n" in line:
            raise ValueError(
                "JSONL line contains newline. Avoid pretty JSON (e.g., indent=2). "
                f"Line index: {idx}."
            )
        lines.append(line)
    content = "\n".join(lines)
    if lines:
        content += "\n"
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(content)


def iter_jsonl(file, max_samples: Optional[int] = None):
    class _JsonlIterable:
        def __init__(self, file_path, max_samples: Optional[int] = None):
            self.file_path = file_path
            self._offsets = None
            self._max_samples = max_samples

        def _ensure_offsets(self):
            if self._offsets is not None:
                return
            offsets = []
            with open(self.file_path, "r") as fp:
                while True:
                    pos = fp.tell()
                    line = fp.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(pos)
            self._offsets = offsets

        def __iter__(self):
            with open(self.file_path, "r") as fp:
                count = 0
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    if self._max_samples is not None and count >= self._max_samples:
                        break
                    count += 1
                    yield json.loads(line)

        def __len__(self):
            self._ensure_offsets()
            if self._max_samples is None:
                return len(self._offsets)
            return min(len(self._offsets), self._max_samples)

        def __getitem__(self, index):
            if not isinstance(index, int):
                raise TypeError("index must be int")
            if index < 0:
                raise IndexError("negative index is not supported")
            self._ensure_offsets()
            effective_len = len(self)
            if index >= effective_len:
                raise IndexError("index out of range")

            with open(self.file_path, "r") as fp:
                fp.seek(self._offsets[index])
                line = fp.readline()
                return json.loads(line)

    return _JsonlIterable(file, max_samples=max_samples)


def load_jsonl(file, max_samples: Optional[int] = None):
    """Load a JSONL file into a list.

    Parameters
    ----------
    file:
        Path to the JSONL file.
    max_samples:
        If given, load at most this many non-empty lines from the file.
    """

    return list(iter_jsonl(file, max_samples=max_samples))


def save_pickle(obj, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)


def save_bytes(data: bytes, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as fp:
        fp.write(data)


def load_bytes(file) -> bytes:
    with open(file, "rb") as fp:
        return fp.read()


__all__ = [
    "save_text",
    "load_text",
    "save_json",
    "load_json",
    "save_jsonl",
    "load_jsonl",
    "iter_jsonl",
    "save_pickle",
    "load_pickle",
    "save_bytes",
    "load_bytes",
]
