# -*- coding: utf-8 -*-
# @Time    : 1/13/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_download.py

import multiprocessing as mp
import os
import tempfile

import pytest

from dl_utils import download

# A small, stable payload for network tests.
TEST_URL = "https://httpbin.org/bytes/1024"


def test_download_reuses_existing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "payload.bin")

        # First download
        downloaded_path = download(TEST_URL, filepath=filepath)

        assert downloaded_path == filepath
        assert os.path.isfile(downloaded_path)
        assert os.path.getsize(downloaded_path) > 0

        # Second download should reuse existing file
        downloaded_path_2 = download(TEST_URL, filepath=filepath)

        assert downloaded_path_2 == filepath
        assert os.path.isfile(downloaded_path_2)

def test_download_with_sha256():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "payload.bin")
        # httpbin's /bytes/<n> endpoint returns random bytes, so it does not have a stable checksum.
        # Fetch once and use the actual sha256 of the downloaded content for the verification path.
        downloaded_path = download(TEST_URL, filepath=filepath)
        import hashlib
        with open(downloaded_path, "rb") as f:
            expected_sha256 = hashlib.sha256(f.read()).hexdigest()

        downloaded_path = download(TEST_URL, filepath=filepath, expected_sha256=expected_sha256)
        assert downloaded_path == filepath

    # Expect error
    with pytest.raises(RuntimeError):
        downloaded_path = download(TEST_URL, filepath=filepath, expected_sha256="error")


def _worker(url, path, queue):
    try:
        result = download(url, filepath=path)
        queue.put(result)
    except Exception as e:
        queue.put(e)


def test_parallel_download_multiprocessing():
    """Test multiple concurrent call of download for the same file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = os.path.join(tmpdir, "payload.bin")

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        processes = [
            ctx.Process(target=_worker, args=(TEST_URL, target_path, queue))
            for _ in range(4)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        results = [queue.get() for _ in processes]

        # All processes should succeed
        for r in results:
            assert isinstance(r, str), f"Worker failed with {r}"

        # All return the same file
        assert all(r == target_path for r in results)

        # File exists and is non-empty
        assert os.path.isfile(target_path)
        assert os.path.getsize(target_path) > 0

        # No temporary files left behind
        leftovers = [
            f for f in os.listdir(tmpdir)
            if f.endswith(".tmp")
        ]
        assert not leftovers
