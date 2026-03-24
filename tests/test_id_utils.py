# -*- coding: utf-8 -*-
# @Time    : 1/28/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_id_utils.py

import re
import tempfile
from pathlib import Path

from dl_utils import generate_id, list_ids


def test_list_ids_with_tempfile():
    """
    Directory layout:

    temp_dir/
        sample_001.txt
        sample_002.txt
        ignore.log
        run_A/
        run_B/
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # --- create files ---
        (root / "sample_001.txt").write_text("data")
        (root / "sample_002.txt").write_text("data")
        (root / "ignore.log").write_text("data")

        # --- create directories ---
        (root / "run_A").mkdir()
        (root / "run_B").mkdir()

        # --- default behavior (auto include -> files, auto matching -> stem) ---
        ids = list_ids(str(root))
        assert sorted(ids) == ["ignore", "sample_001", "sample_002"]

        # --- include only .txt files ---
        ids = list_ids(
            str(root),
            include=r"\.txt$",
        )
        assert sorted(ids) == ["sample_001", "sample_002"]

        # --- extract numeric IDs via regex ---
        ids = list_ids(
            str(root),
            include=r"\.txt$",
            matching=re.compile(r"_(\d+)\.txt$"),
        )
        assert sorted(ids) == ["001", "002"]

        # --- exclude pattern ---
        ids = list_ids(
            str(root),
            exclude=r"ignore",
        )
        assert "ignore" not in ids


def test_list_ids_directories_only_with_tempfile():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # more dirs than files
        (root / "A").mkdir()
        (root / "B").mkdir()
        (root / "C").mkdir()
        (root / "file.txt").write_text("x")

        ids = list_ids(str(root))
        assert sorted(ids) == ["A", "B", "C"]


def test_list_ids_return_filepaths_with_tempfile():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        (root / "x_01.txt").write_text("x")
        (root / "x_02.txt").write_text("x")

        ids, paths = list_ids(
            str(root),
            matching=r"x_(\d+)",
            return_filepath=True,
        )

        assert sorted(ids) == ["01", "02"]
        assert len(ids) == len(paths)
        assert all(p.endswith(".txt") for p in paths)

        # --- return dict ---
        id2path = list_ids(
            str(root),
            matching=r"x_(\d+)",
            return_filepath=True,
            return_dict=True,
        )
        print(id2path)

        assert sorted(id2path.keys()) == ["01", "02"]
        assert len(id2path) == len(ids)
        assert all(p.endswith(".txt") for p in id2path.values())


def test_generate_id_random_mode():
    # --- random mode ---
    ids = [generate_id() for _ in range(100)]
    print(f"Random ids {ids}")
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[0-9a-f]{32}", x) for x in ids)


def test_generate_id_deterministic_mode():
    # --- deterministic mode ---
    a1 = generate_id("hello")
    a2 = generate_id("hello")
    b = generate_id("world")

    print(f"{a1}=={a2} {a1}!={b}")
    assert a1 == a2
    assert a1 != b
    assert re.fullmatch(r"[0-9a-f]{32}", a1)
