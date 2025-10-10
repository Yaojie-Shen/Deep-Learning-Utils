# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : text.py

from pathlib import Path


def save_text(text: str, file):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(text)


def load_text(file) -> str:
    with open(file, "r") as fp:
        return fp.read()


__all__ = ["save_text", "load_text"]
