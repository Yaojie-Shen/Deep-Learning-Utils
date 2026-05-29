# -*- coding: utf-8 -*-
# @Time    : 1/20/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_save_and_load_utils.py
import json
import tempfile
from pathlib import Path

from tqdm import tqdm

from dl_utils import (concurrent_file_loader, iter_files, iter_jsonl,
                      load_bytes, load_files, load_json, load_jsonl,
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


def test_concurrent_file_loader_large_scale():
    num_files = 150

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        file_paths = []
        for i in range(num_files):
            path = base / f"file_{i:03d}.json"
            save_json({"index": i, "content": f"data_{i}"}, path)
            file_paths.append(path)

        results = list(
            concurrent_file_loader(
                file_paths,
                loader=load_json,
                concurrency_limit=2,
                chunk_size=10,
            )
        )

        assert len(results) == num_files
        for i, data in enumerate(results):
            assert data["index"] == i
            assert data["content"] == f"data_{i}"


def test_iter_and_load_files_from_file_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        first = base / "first.json"
        second = base / "second.json"
        save_json([{"id": 1}, {"id": 2}], first)
        save_json({"id": 3}, second)

        assert list(iter_files([second, first], pattern="*.json", loader=load_json, n_jobs=2)) == [
            {"id": 3},
            [{"id": 1}, {"id": 2}],
        ]
        assert load_files([first, second], pattern="*.json", flatten=True, loader=load_json, n_jobs=2) == [
            {"id": 1},
            {"id": 2},
            {"id": 3},
        ]


def test_iter_and_load_files_file_list_without_pattern():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        first = base / "first.json"
        second = base / "second.json"
        save_json({"id": 1}, first)
        save_json({"id": 2}, second)

        assert list(iter_files([second, first], loader=load_json, n_jobs=2)) == [
            {"id": 2},
            {"id": 1},
        ]
        assert load_files([first, second], loader=load_json, n_jobs=2) == [
            {"id": 1},
            {"id": 2},
        ]


def test_iter_and_load_files_default_to_bytes_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        first = base / "first.bin"
        second = base / "second.bin"
        save_bytes(b"first", first)
        save_bytes(b"second", second)

        assert list(iter_files([second, first], n_jobs=2)) == [b"second", b"first"]
        assert load_files([first, second], n_jobs=2) == [b"first", b"second"]


def test_load_files_directory_requires_pattern():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        save_json({"id": 1}, base / "data.json")

        try:
            load_files(base, n_jobs=2)
            assert False, "Expected ValueError when loading a directory without pattern"
        except ValueError as exc:
            assert "pattern" in str(exc)


def test_load_files_from_directory_sorted_and_recursive_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        nested = base / "nested"
        nested.mkdir()
        save_json({"name": "b"}, base / "b.json")
        save_json({"name": "a"}, base / "a.json")
        save_json({"name": "nested"}, nested / "c.json")
        save_text("ignore", base / "ignore.txt")

        assert load_files(base, pattern="*.json", loader=load_json, n_jobs=2) == [
            {"name": "a"},
            {"name": "b"},
        ]
        assert load_files(base, pattern="**/*.json", loader=load_json, n_jobs=2) == [
            {"name": "a"},
            {"name": "b"},
            {"name": "nested"},
        ]


def test_load_files_accepts_loader_function():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "b.txt").write_text("B", encoding="utf-8")
        (base / "a.txt").write_text("A", encoding="utf-8")

        assert load_files(base, pattern="*.txt", loader=load_text, n_jobs=2) == ["A", "B"]


def test_iter_files_accepts_custom_loader_and_kwargs():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        first = base / "1.txt"
        second = base / "2.txt"
        first.write_text("hello", encoding="utf-8")
        second.write_text("world", encoding="utf-8")

        def load_with_prefix(path, prefix=""):
            return f"{prefix}{Path(path).read_text(encoding='utf-8')}"

        assert list(
            iter_files(
                [first, second],
                pattern="*.txt",
                loader=load_with_prefix,
                load_kwargs={"prefix": "say-"},
                n_jobs=2,
            )
        ) == ["say-hello", "say-world"]


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


def test_save_jsonl_accepts_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.jsonl"

        save_jsonl(({"i": i} for i in range(3)), str(f))

        assert load_jsonl(str(f)) == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_save_jsonl_writes_json_array_for_json_suffix():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.json"

        save_jsonl(({"i": i, "b": b"bytes"} for i in range(3)), str(f))

        content = f.read_text()
        assert content.startswith("[\n")
        assert not content.rstrip().endswith(",")
        assert json.loads(content) == [
            {"i": 0, "b": "bytes"},
            {"i": 1, "b": "bytes"},
            {"i": 2, "b": "bytes"},
        ]


def test_load_jsonl_max_samples():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.jsonl"

        data = [{"i": i} for i in range(5)]
        save_jsonl(data, str(f))

        loaded = load_jsonl(str(f), max_samples=2)

        assert len(loaded) == 2
        assert loaded == data[:2]


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


def test_iter_jsonl_max_samples_affects_len_and_iteration():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        f = base / "data.jsonl"

        f.write_text("""{"a": 1}
{"a": 2}
{"a": 3}
""")

        it = iter_jsonl(str(f), max_samples=2)
        assert len(it) == 2
        assert list(it) == [{"a": 1}, {"a": 2}]
        assert it[0] == {"a": 1}
        try:
            _ = it[2]
            assert False, "Expected IndexError when accessing beyond max_samples"
        except IndexError:
            pass


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
