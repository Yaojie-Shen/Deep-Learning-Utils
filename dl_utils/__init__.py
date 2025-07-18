# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 18:52
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : __init__.py

from typing import TYPE_CHECKING

from .import_utils import _LazyModule, define_import_structure

if TYPE_CHECKING:
    from .data import *
    from .distributed import *
    from .fs import *
    from .visualize import *

    from .decorators import *
    from .prefetcher import *
    from .qps_control import *
    from .timer import *
    from .type_hint import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
