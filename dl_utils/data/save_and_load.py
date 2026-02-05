# -*- coding: utf-8 -*-
# @Time    : 1/19/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : save_and_load.py

import json
import pickle
from pathlib import Path


def save_text(text: str, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(text)


def load_text(file) -> str:
    with open(file, "r") as fp:
        return fp.read()


def save_json(data, file, save_pretty=False, **kwargs):
    """Save json to file.

    - If `save_pretty=True`, write a human-readable JSON string using `json.dumps` with `indent=4` and `sort_keys=True`.
    - Extra keyword arguments forwarded to `json.dump`.
    """

    class MyEncoder(json.JSONEncoder):
        """Object of type bytes is not JSON serializable, convert it to string before saving"""

        def default(self, obj):
            if isinstance(obj, bytes):  # bytes->str
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    _kwargs = {"cls": MyEncoder}
    if save_pretty:
        _kwargs.update({"indent": 4, "ensure_ascii": False})
    _kwargs.update(kwargs)

    # Dump to string first to avoid writing if error occurs
    s = json.dumps(data, **_kwargs)
    with open(file, "w") as fp:
        fp.write(s)


def load_json(file):
    with open(file, "r") as fp:
        return json.load(fp)


def save_pickle(obj, file):
    with open(file, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pickle(file):
    with open(file, 'rb') as fp:
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
    "save_pickle",
    "load_pickle",
    "save_bytes",
    "load_bytes",
]
