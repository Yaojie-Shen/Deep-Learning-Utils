# -*- coding: utf-8 -*-
# @Time    : 3/6/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : env_utils.py

import os
from pathlib import Path


def _normalize_newlines(value: str, allow_multiline: bool) -> str:
    """Normalize or strip newlines from a string.

    - If allow_multiline is False, all line breaks ("\r", "\n") are removed so a single-line string is returned.
    - If allow_multiline is True, preserve multiple lines but normalize to "\n" line endings.
    """
    if allow_multiline:
        # Normalize Windows/Mac newlines to \n for consistency
        return value.replace("\r\n", "\n").replace("\r", "\n")
    else:
        # Remove all line breaks entirely to return a single line string
        return "".join(value.splitlines())


def get_env(name: str, allow_multiline: bool = False, cwd: str | Path | None = None) -> str:
    """Get a secret or config value from environment or a dotfile in the working directory.

    Lookup order (later overwrites earlier if both exist):
    1) Environment variable `name`.
    2) A file named `.{name}` under the current working directory (or `cwd` if provided).

    By default, the returned string contains no line breaks. Set `allow_multiline=True` to preserve
    multiple lines (line endings normalized to ``\n``).

    Args:
        name: Environment variable name, and the filename stem for the `.{name}` file.
        allow_multiline: Whether to preserve multiple lines in the returned string.
        cwd: Optional directory to resolve the `.{name}` file from. Defaults to `Path.cwd()`.

    Returns:
        The resolved secret/config value as a string.

    Raises:
        KeyError: If the value is not found in either the environment or the `.{name}` file.
    """

    # 1) Read from environment
    chosen: str | None = os.environ.get(name)

    # 2) Read from ./{name} file and overwrite if present
    workdir = Path(cwd) if cwd is not None else Path.cwd()
    file_path = workdir / f".{name}"
    if file_path.exists() and file_path.is_file():
        file_text = file_path.read_text(encoding="utf-8")
        chosen = file_text

    if chosen is None:
        raise KeyError(
            f"'{name}' not found in environment or file '{file_path}'. "
            f"Set env var {name} or create a file '{file_path.name}' in {workdir}."
        )

    return _normalize_newlines(chosen, allow_multiline=allow_multiline)


__all__ = [
    "get_env",
]

