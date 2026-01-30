# -*- coding: utf-8 -*-
# @Time    : 1/20/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_save_and_load_utils.py
# python
import tempfile
from pathlib import Path

from dl_utils import (
    save_text,
    load_text,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_bytes,
    load_bytes
)


def test_save_and_load_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "sub" / "hello.txt"

        save_text("hello world", str(f))

        assert f.exists()
        assert load_text(str(f)) == "hello world"

        save_text("hello again\n\n\n", str(f))
        assert load_text(str(f)) == "hello again\n\n\n"


def test_save_and_load_json_with_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.json"

        data = {"num": 1, "lst": [1, 2, 3], "b": b"bytes"}
        save_json(data, str(f))
        loaded = load_json(str(f))

        assert loaded["num"] == 1
        assert loaded["lst"] == [1, 2, 3]
        # bytes should be converted to a utf-8 string by the custom encoder
        assert isinstance(loaded["b"], str)
        assert loaded["b"] == "bytes"


def test_save_json_pretty_and_sorted_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "pretty.json"

        data = {"z": 2, "a": 1}
        save_json(data, str(f), save_pretty=True)

        content = f.read_text()

        # pretty option should produce newlines and indentation
        assert "\n" in content
        assert "    " in content or "\t" in content

        # sort_keys=True when save_pretty -> "a" should appear before "z"
        assert content.index('"a"') < content.index('"z"')


def test_save_and_load_pickle():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "obj.pkl"

        obj = {"k": (1, 2), "s": "x"}
        save_pickle(obj, str(f))
        loaded = load_pickle(str(f))

        assert loaded == obj


def test_save_and_load_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.bin"

        data = b"\x00\x01\x02\x03\x04"
        save_bytes(data, str(f))
        loaded = load_bytes(str(f))

        assert loaded == data
