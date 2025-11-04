"""Inference cache shared by GPU workers."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Awaitable, Callable, Dict, Optional

from .models import CachedConditions


class InferenceCache:
    """Thread-safe cache for pre-processed conditioning tensors."""

    def __init__(self) -> None:
        self._store: Dict[str, CachedConditions] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    @staticmethod
    def make_key(image_path: str, mode: str) -> str:
        digest = hashlib.sha1(
            f"{image_path}:{mode}".encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        return digest

    async def get(self, key: str) -> Optional[CachedConditions]:
        async with self._global_lock:
            return self._store.get(key)

    async def get_or_create(
        self,
        key: str,
        factory: Callable[[], Awaitable[CachedConditions]],
    ) -> CachedConditions:
        async with self._global_lock:
            cached = self._store.get(key)
            if cached is not None:
                return cached
            lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            async with self._global_lock:
                cached = self._store.get(key)
                if cached is not None:
                    return cached

            value = await factory()

            async with self._global_lock:
                self._store[key] = value
                self._locks.pop(key, None)
            return value
