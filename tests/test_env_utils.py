# -*- coding: utf-8 -*-
# @Time    : 3/6/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_env_utils.py

import tempfile
from pathlib import Path

import pytest

from dl_utils.env_utils import get_env


def test_get_env_from_env_when_no_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        # Ensure file does not exist
        target = workdir / ".API_TOKEN"
        assert not target.exists()

        # Set env var only
        monkeypatch.setenv("API_TOKEN", "line1\r\nline2\rline3\n")

        # Default: single-line result (strip all line breaks)
        assert get_env("API_TOKEN", cwd=workdir) == "line1line2line3"

        # Preserve multiple lines and normalize to \n
        assert get_env("API_TOKEN", allow_multiline=True, cwd=workdir) == "line1\nline2\nline3\n"


def test_get_env_prefers_file_over_env(monkeypatch):
    """
    Directory layout:

    tmp/
        .MYSECRET  (file content should override environment)
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        # Set env var first
        monkeypatch.setenv("MYSECRET", "env-value")

        # Create the dotfile which should take precedence
        (workdir / ".MYSECRET").write_text("file\nvalue\r\nline2", encoding="utf-8")

        # Multiline preserved and normalized
        assert get_env("MYSECRET", allow_multiline=True, cwd=workdir) == "file\nvalue\nline2"

        # Single-line (no line breaks)
        assert get_env("MYSECRET", allow_multiline=False, cwd=workdir) == "filevalueline2"


def test_get_env_missing_raises(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        # Make sure env is absent and no file exists
        monkeypatch.delenv("NOT_EXISTS_KEY", raising=False)
        assert not (workdir / ".NOT_EXISTS_KEY").exists()

        with pytest.raises(KeyError):
            _ = get_env("NOT_EXISTS_KEY", cwd=workdir)


def test_get_env_cwd_resolution(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        # No env var set
        monkeypatch.delenv("FOO", raising=False)

        # Read from provided cwd
        (workdir / ".FOO").write_text("bar\n", encoding="utf-8")

        assert get_env("FOO", cwd=workdir) == "bar"
        assert get_env("FOO", allow_multiline=True, cwd=workdir) == "bar\n"

