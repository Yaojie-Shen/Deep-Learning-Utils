# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_ollama_utils.py

import asyncio
import time

from dl_utils import OllamaModel, ensure_ollama_running, inspect_data


def test_gemini_chat():
    ollama_model = OllamaModel(model_name="gemma3:270m")

    resp = ollama_model.chat(messages=[{"role": "user", "content": "1+1=?"}])
    inspect_data(dict(resp), max_dict_items=20)


def test_ensure_ollama_running_cost():
    for _ in range(1000):
        s_time = time.time()
        ensure_ollama_running()
        print(f"{time.time() - s_time}")


def test_async_chat():
    ollama_model = OllamaModel(model_name="gemma3:270m")

    async def async_chat():
        resp = await ollama_model.async_chat(
            messages=[{"role": "user", "content": "1+1=?"}]
        )
        inspect_data(dict(resp), max_dict_items=20)
        print(resp.message.content)

    asyncio.run(async_chat())
