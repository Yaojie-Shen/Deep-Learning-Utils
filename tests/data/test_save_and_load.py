# -*- coding: utf-8 -*-
# @Time    : 1/20/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_save_and_load_utils.py
import tempfile
from pathlib import Path

from tqdm import tqdm

from dl_utils import (iter_jsonl, load_bytes, load_json, load_jsonl,
                      load_pickle, load_text, save_bytes, save_json,
                      save_jsonl, save_pickle, save_text)


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


def test_save_json_pretty():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "pretty.json"

        data = {"z": 2, "a": 1}
        save_json(data, str(f), save_pretty=True)

        content = f.read_text()

        # pretty option should produce newlines and indentation
        assert "\n" in content
        assert "    " in content or "\t" in content


def test_save_and_load_jsonl_with_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.jsonl"

        data = [{"num": 1, "b": b"bytes"}, {"num": 2, "lst": [1, 2, 3]}]
        save_jsonl(data, str(f))
        loaded = load_jsonl(str(f))

        assert loaded[0]["num"] == 1
        assert loaded[0]["b"] == "bytes"
        assert loaded[1]["num"] == 2
        assert loaded[1]["lst"] == [1, 2, 3]


def test_save_jsonl_rejects_newlines():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.jsonl"

        try:
            save_jsonl([{"a": 1}], str(f), indent=2)
            assert False, "Expected ValueError due to newline in JSONL line"
        except ValueError as exc:
            assert "indent" in str(exc)


def test_iter_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.jsonl"

        f.write_text('{"a": 1}\n\n{"a": 2}\n')

        it = iter_jsonl(str(f))
        assert len(it) == 2
        assert it[0] == {"a": 1}
        assert it[1] == {"a": 2}
        assert list(it) == [{"a": 1}, {"a": 2}]

        tqdm_items = list(tqdm(iter_jsonl(str(f))))
        assert tqdm_items == [{"a": 1}, {"a": 2}]


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
