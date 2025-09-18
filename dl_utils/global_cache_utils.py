# -*- coding: utf-8 -*-
# @Time    : 9/12/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : global_cache_utils.py

from collections import OrderedDict


class GlobalCache:
    """Create a singleton key-value cache."""

    _instance = None  # singleton instance

    def __new__(cls, max_size=None):
        """Return the singleton instance"""
        if cls._instance is None:
            cls._instance = super(GlobalCache, cls).__new__(cls)
            cls._instance.max_size = max_size
            cls._instance.cache = OrderedDict()
        return cls._instance

    def get(self, key):
        """Get cached value; if exists, update its usage order"""
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value  # Move to the end (most recently used)
            return value
        return None

    def set(self, key, value):
        """Set cached value, supports LRU eviction"""
        if key in self.cache:
            self.cache.pop(key)
        elif self.max_size and len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Evict the oldest item
        self.cache[key] = value

    def clear(self):
        """Clear all cached items"""
        self.cache.clear()


__all__ = [
    "GlobalCache"
]
