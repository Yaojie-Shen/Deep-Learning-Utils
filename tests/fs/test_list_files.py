# -*- coding: utf-8 -*-
# @Time    : 5/23/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_list_files.py
import time

from pytest import mark, param, main
from tqdm import tqdm

from dl_utils import list_files, list_files_multithread


@mark.parametrize(
    "root,depth",
    [
        param(".", None),
        param(".", 1),
    ]
)
def test_list_files(root: str, depth: int):
    s_time = time.time()
    files = list(tqdm(list_files(root, depth=depth), desc="single thread"))
    print(f"Total files: {len(files)}")
    print(f"`list_files` took {time.time() - s_time}s")

    s_time = time.time()
    files_mt = list(tqdm(list_files_multithread(root, depth=depth), desc="multi-thread"))
    print(f"`list_files_multithread` took {time.time() - s_time}s")

    assert len(files) == len(set(files))
    assert set(files) == set(files_mt)

    print(files)


if __name__ == '__main__':
    main()
