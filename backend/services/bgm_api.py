"""Bangumi API v0 client with rate limiting and SQLite caching."""

import asyncio
import json
import time
import sqlite3
from pathlib import Path

import httpx

from backend.config import (
    BGM_API_BASE,
    BGM_API_RATE_LIMIT,
    CACHE_DB_PATH,
    CACHE_TTL_USER_COLLECTION,
    CACHE_TTL_SUBJECT_DETAIL,
)


class BangumiAPIClient:
    def __init__(self):
        self.base_url = BGM_API_BASE
        self.semaphore = asyncio.Semaphore(BGM_API_RATE_LIMIT)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30,
            headers={
                "User-Agent": "BangumiRecommender/1.0 (https://github.com/bangumi)",
                "Accept": "application/json",
            },
        )
        self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache database."""
        CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def _cache_get(self, key: str) -> dict | None:
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        row = conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row and row[1] > time.time():
            return json.loads(row[0])
        return None

    def _cache_set(self, key: str, value, ttl: int):
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), time.time() + ttl),
        )
        conn.commit()
        conn.close()

    async def _rate_limited_get(self, path: str, params: dict | None = None):
        async with self.semaphore:
            resp = await self.client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_user_collections(
        self, username: str, subject_type: int = 2
    ) -> list[dict]:
        """Fetch all collections for a user, with pagination and caching."""
        cache_key = f"user_collections:{username}:{subject_type}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        all_items = []
        offset = 0
        limit = 50

        while True:
            try:
                data = await self._rate_limited_get(
                    f"/users/{username}/collections",
                    params={
                        "subject_type": subject_type,
                        "limit": limit,
                        "offset": offset,
                    },
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return []
                raise

            items = data.get("data", [])
            if not items:
                break

            all_items.extend(items)
            total = data.get("total", 0)

            offset += limit
            if offset >= total:
                break

        self._cache_set(cache_key, all_items, CACHE_TTL_USER_COLLECTION)
        return all_items

    async def get_subject(self, subject_id: int) -> dict | None:
        """Fetch subject details with caching."""
        cache_key = f"subject:{subject_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            data = await self._rate_limited_get(f"/subjects/{subject_id}")
            self._cache_set(cache_key, data, CACHE_TTL_SUBJECT_DETAIL)
            return data
        except httpx.HTTPStatusError:
            return None

    async def close(self):
        await self.client.aclose()
