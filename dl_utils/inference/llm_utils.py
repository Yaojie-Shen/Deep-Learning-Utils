# -*- coding: utf-8 -*-
# @Time    : 2/6/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : llm_utils.py

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

PathLike = Union[str, os.PathLike]


def _normalize_extensions(extensions: Sequence[str]) -> tuple:
    return tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)


def _is_file_path(value: PathLike) -> bool:
    try:
        return Path(value).expanduser().is_file()
    except OSError:
        return False


def _looks_like_prompt_path(value: PathLike, extensions: Sequence[str]) -> bool:
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str):
        return False

    normalized_extensions = _normalize_extensions(extensions)
    suffix = Path(value).suffix
    has_path_separator = os.sep in value or (
        os.altsep is not None and os.altsep in value
    )
    return suffix in normalized_extensions or has_path_separator


def _resolve_prompt_path(
    prompt_dir: PathLike,
    prompt_name: str,
    version: Optional[str] = None,
    extensions: Sequence[str] = (".txt", ".md"),
) -> Path:
    prompt_dir = Path(prompt_dir).expanduser()
    prompt_name_path = Path(prompt_name)

    if prompt_name_path.suffix:
        candidates = [prompt_dir / prompt_name_path]
    else:
        normalized_extensions = _normalize_extensions(extensions)
        candidates = []

        if version is not None:
            version = str(version)
            for ext in normalized_extensions:
                candidates.extend(
                    [
                        prompt_dir / f"{prompt_name}_{version}{ext}",
                        prompt_dir / f"{prompt_name}.{version}{ext}",
                        prompt_dir / prompt_name / f"{version}{ext}",
                    ]
                )
        else:
            candidates.extend(
                prompt_dir / f"{prompt_name}{ext}" for ext in normalized_extensions
            )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot find prompt file. Tried: {candidate_text}")


def load_prompt(
    prompt: Optional[PathLike] = None,
    *,
    prompt_dir: Optional[PathLike] = None,
    prompt_name: Optional[str] = None,
    version: Optional[str] = None,
    extensions: Sequence[str] = (".txt", ".md"),
    encoding: str = "utf-8",
    input_type: str = "auto",
) -> str:
    """
    Load prompt text from a file path, a prompt directory, or return text directly.

    Args:
        prompt: Prompt file path or already-loaded prompt text. In ``auto`` mode,
            existing paths are loaded from disk, strings that look like prompt
            paths raise ``FileNotFoundError`` if missing, and other strings are
            treated as text. Use ``input_type="text"`` to force a path-like
            string to be treated as literal prompt text.
        prompt_dir: Directory that stores prompt files.
        prompt_name: Prompt name under ``prompt_dir``. If it has no suffix,
            ``extensions`` are tried in order. When ``version`` is set, common
            versioned names are tried, such as ``name_v1.txt``, ``name.v1.txt``
            and ``name/v1.txt``.
        version: Optional prompt version saved as a separate file.
        extensions: Candidate prompt file extensions. Defaults to ``.txt`` and
            ``.md``.
        encoding: File encoding used when reading prompt files.
        input_type: One of ``auto``, ``path`` or ``text``.

    Returns:
        Loaded prompt text.
    """
    if input_type not in {"auto", "path", "text"}:
        raise ValueError("input_type must be one of: 'auto', 'path', 'text'.")

    if prompt_dir is not None or prompt_name is not None:
        if prompt_dir is None or prompt_name is None:
            raise ValueError("prompt_dir and prompt_name must be provided together.")
        path = _resolve_prompt_path(prompt_dir, prompt_name, version, extensions)
        return path.read_text(encoding=encoding)

    if prompt is None:
        raise ValueError(
            "Either prompt or both prompt_dir and prompt_name must be provided."
        )

    if input_type == "text":
        return str(prompt)

    if input_type == "path" or _is_file_path(prompt):
        return Path(prompt).expanduser().read_text(encoding=encoding)

    if _looks_like_prompt_path(prompt, extensions):
        raise FileNotFoundError(f"Prompt path does not exist: {prompt}")

    return str(prompt)


def _format_bracket_prompt(prompt: str, variables: Mapping[str, Any]) -> str:
    pattern = re.compile(r"(?<!\[)\[([A-Za-z_]\w*)\](?!\])")

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(key)
        return str(variables[key])

    return pattern.sub(replace, prompt).replace("[[", "[").replace("]]", "]")


def format_prompt(
    prompt: str,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    style: Optional[str] = "format",
    **kwargs: Any,
) -> str:
    """
    Fill placeholders in a prompt string.

    Args:
        prompt: Prompt text to format.
        variables: Placeholder values.
        style: Placeholder style. ``format``/``brace``/``python`` uses Python
            ``str.format`` syntax like ``{name}`` and escapes literal braces as
            ``{{text}}``. ``bracket``/``square`` uses ``[name]`` and escapes
            literal brackets as ``[[text]]``. ``none`` disables formatting.
        **kwargs: Extra placeholder values. Values here override ``variables``.

    Returns:
        Formatted prompt text.
    """
    data = dict(variables or {})
    data.update(kwargs)

    if style is None or style == "none":
        return prompt

    if style in {"format", "brace", "python"}:
        return prompt.format_map(data)

    if style in {"bracket", "square"}:
        return _format_bracket_prompt(prompt, data)

    raise ValueError(
        "style must be one of: 'format', 'brace', 'python', 'bracket', 'square', 'none'."
    )


def render_prompt(
    prompt: Optional[PathLike] = None,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    prompt_dir: Optional[PathLike] = None,
    prompt_name: Optional[str] = None,
    version: Optional[str] = None,
    extensions: Sequence[str] = (".txt", ".md"),
    encoding: str = "utf-8",
    input_type: str = "auto",
    style: Optional[str] = "format",
    **kwargs: Any,
) -> str:
    """
    Load a prompt from path/text and fill placeholders in one call.

    Args:
        prompt: Prompt file path or already-loaded prompt text.
        variables: Placeholder values.
        prompt_dir: Directory that stores prompt files.
        prompt_name: Prompt name under ``prompt_dir``.
        version: Optional prompt version saved as a separate file.
        extensions: Candidate prompt file extensions.
        encoding: File encoding used when reading prompt files.
        input_type: One of ``auto``, ``path`` or ``text``.
        style: Placeholder style passed to :func:`format_prompt`.
        **kwargs: Extra placeholder values. Values here override ``variables``.

    Returns:
        Rendered prompt text.
    """
    prompt_text = load_prompt(
        prompt,
        prompt_dir=prompt_dir,
        prompt_name=prompt_name,
        version=version,
        extensions=extensions,
        encoding=encoding,
        input_type=input_type,
    )
    return format_prompt(prompt_text, variables, style=style, **kwargs)


def extract_json(llm_output: str) -> Any:
    """
    Extract and parse JSON from an LLM output string.
    The output may contain text before or after the JSON, including markdown code fences. Returns the parsed Python object.

    Raises:
        ValueError: if no valid JSON can be found or parsed.
    """
    # 1. Try to extract JSON inside Markdown code fences first
    fence_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\}|$begin:math:display$\.\*\?$end:math:display$)\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    fence_match = fence_pattern.search(llm_output)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to more general extraction

    # 2. Fallback: find the first valid JSON object or array by scanning
    start_indices = [i for i, ch in enumerate(llm_output) if ch in "{["]

    for start in start_indices:
        for end in range(len(llm_output), start, -1):
            if llm_output[end - 1] not in "}]":
                continue
            candidate = llm_output[start:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found in LLM output.")


__all__ = [
    "extract_json",
    "format_prompt",
    "load_prompt",
    "render_prompt",
]
