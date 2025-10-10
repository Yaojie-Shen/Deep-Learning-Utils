# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : pickle.py

import pickle


def save_pickle(obj, file):
    with open(file, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pickle(file):
    with open(file, 'rb') as fp:
        return pickle.load(fp)


__all__ = ["save_pickle", "load_pickle"]
