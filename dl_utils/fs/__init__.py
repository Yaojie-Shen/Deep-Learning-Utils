# -*- coding: utf-8 -*-
# @Time    : 7/16/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : __init__.py

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule, define_import_structure

if TYPE_CHECKING:
    from .list_files import *
    from .dir import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
