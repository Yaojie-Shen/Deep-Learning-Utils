# -*- coding: utf-8 -*-
# @Time    : 4/20/23
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : json.py

import json


def load_json(file):
    with open(file, "r") as fp:
        return json.load(fp)


def save_json(data, file, save_pretty=False, sort_keys=False):
    class MyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, bytes):  # bytes->str
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    with open(file, "w") as fp:
        if save_pretty:
            fp.write(json.dumps(data, cls=MyEncoder, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, fp)


__all__ = ["load_json", "save_json"]
