# -*- coding: utf-8 -*-
# @Time    : 2/6/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : llm_utils.py

import json
import re
from typing import Any


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
    start_indices = [
        i for i, ch in enumerate(llm_output) if ch in "{["
    ]

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
]
