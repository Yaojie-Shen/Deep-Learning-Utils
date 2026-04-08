# -*- coding: utf-8 -*-
# @Time    : 2/6/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_llm_utils.py

import pytest

from dl_utils import extract_json


@pytest.mark.parametrize(
    "llm_output, expected",
    [
        # Plain JSON
        ('{"key": "value"}', {"key": "value"}),
        # JSON in markdown code fences
        ("```json\n{\"key\": 123}\n```", {"key": 123}),
        # JSON with text before and after
        ("Here is the data:\n```json\n{\"foo\": \"bar\"}\n```\nThank you!", {"foo": "bar"}),
        # JSON array
        ('[1, 2, 3]', [1, 2, 3]),
        # JSON with extra text but valid extraction
        ("Output: {\"a\": true, \"b\": false} End", {"a": True, "b": False}),
    ]
)
def test_extract_json_valid(llm_output, expected):
    assert extract_json(llm_output) == expected


def test_extract_json_invalid():
    # Invalid JSON should raise ValueError
    with pytest.raises(ValueError):
        extract_json("No JSON here!")


def test_extract_json_partial_malformed():
    # Malformed JSON inside text, but valid part exists
    llm_output = "Data: {\"valid\": 1, \"invalid\": } end"
    # Should raise ValueError because no fully valid JSON can be parsed
    with pytest.raises(ValueError):
        extract_json(llm_output)
