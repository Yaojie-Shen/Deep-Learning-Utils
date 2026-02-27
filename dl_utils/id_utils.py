# -*- coding: utf-8 -*-
# @Time    : 1/28/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : id_utils.py
import re
from pathlib import Path
from typing import Iterable


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _compile_patterns(patterns):
    compiled = []
    for p in _as_list(patterns):
        if isinstance(p, re.Pattern):
            compiled.append(p)
        else:
            compiled.append(re.compile(p))
    return compiled


def _matches_any(name: str, patterns: Iterable[re.Pattern]) -> bool:
    return any(p.search(name) for p in patterns)


def list_ids(
        roots: str | list[str],
        include: str | re.Pattern | list[re.Pattern] = "auto",
        exclude: str | re.Pattern | list[re.Pattern] = None,
        matching: str | re.Pattern = "auto",
        return_filepath: bool = False,
        return_dict: bool = False,
) -> list[str] | tuple[list[str], list[str]] | dict[str, str]:
    """
    Extract sample IDs from file or folder names under given root paths.

    Args:
        roots: Root path(s) to search for files or folders.
        include: Patterns to include files/folders in directory. If "auto", include all files if there are more files than folders, else include all folders.
        exclude: Patterns to exclude files/folders in directory.
        matching: Patterns to extract IDs from file/folder names. If "auto", use common patterns: for files use the full name without extension, for folders use the full folder name.
        return_filepath: Whether to return the full file/folder paths along with IDs.
        return_dict: Whether to return a dictionary mapping IDs to file/folder paths. Only effective when return_filepath is True.

    Returns: A list of extracted IDs, or a tuple of (IDs, file/folder paths) if return_filepath is True.
    """
    roots = [Path(r) for r in _as_list(roots)]
    exclude_patterns = _compile_patterns(exclude)

    ids: list[str] = []
    paths: list[str] = []

    for root in roots:
        if not root.exists():
            continue

        entries = list(root.iterdir())
        if not entries:
            continue

        files = [e for e in entries if e.is_file()]
        dirs = [e for e in entries if e.is_dir()]

        # --- auto include logic ---
        if include == "auto":
            candidates = files if len(files) >= len(dirs) else dirs
            include_patterns = []
        else:
            candidates = entries
            include_patterns = _compile_patterns(include)

        for item in candidates:
            name = item.name

            # include filter
            if include_patterns and not _matches_any(name, include_patterns):
                continue

            # exclude filter
            if exclude_patterns and _matches_any(name, exclude_patterns):
                continue

            # --- ID extraction ---
            if matching == "auto":
                if item.is_file():
                    sample_id = item.stem
                else:
                    sample_id = name
            else:
                pattern = (
                    matching
                    if isinstance(matching, re.Pattern)
                    else re.compile(matching)
                )
                m = pattern.search(name)
                if not m:
                    continue
                sample_id = m.group(1) if m.groups() else m.group(0)

            ids.append(sample_id)
            paths.append(str(item.resolve()))

    if return_dict:
        return {id_: path for id_, path in zip(ids, paths)}

    return (ids, paths) if return_filepath else ids


__all__ = [
    "list_ids"
]
