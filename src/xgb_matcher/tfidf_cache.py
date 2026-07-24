"""Persistent TF-IDF cache per partition.

Serializes (vectorizer, matrix, names, char_vec, char_mat, addr_vec, addr_mat)
to disk so they survive across runs.  Invalidated by RetrievalConfigV1 signature
hash — if retrieval params change, the cache is rebuilt automatically.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_logger = logging.getLogger(__name__)

# Default cache root (overridable via XGB_TFIDF_CACHE_DIR env var)
_DEFAULT_CACHE_DIR = Path("data/tfidf_cache")


def _cache_root() -> Path:
    return Path(os.getenv("XGB_TFIDF_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))


# Type alias for the full TF-IDF artifact bundle per partition
TfidfArtifacts = Tuple[Any, Any, Any, Any, Any, Any, Any]
# (name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat)


class TfidfPersistentCache:
    """Disk-backed cache for TF-IDF artifacts keyed by partition + config hash.

    Directory layout:
        <cache_root>/<config_hash>/<partition_key>.pkl

    Usage:
        cache = TfidfPersistentCache(config_hash="abc123")
        cached = cache.get("insee_75056")
        if cached is None:
            artifacts = _build_all_tfidf_indexes(...)
            cache.put("insee_75056", artifacts)
    """

    def __init__(
        self,
        config_hash: str,
        cache_dir: Path | None = None,
        *,
        fallback_config_hashes: list[str] | None = None,
    ):
        self.cache_dir = (cache_dir or _cache_root()) / config_hash
        self.fallback_dirs = [
            (cache_dir or _cache_root()) / item
            for item in (fallback_config_hashes or [])
            if item and item != config_hash
        ]
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _safe_key(partition_key: str) -> str:
        return (
            partition_key.replace("|", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    def _key_path(self, partition_key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(partition_key)}.pkl"

    def get(self, partition_key: str) -> Optional[TfidfArtifacts]:
        """Load cached artifacts. Returns None on miss or corruption."""
        paths = [self._key_path(partition_key)] + [
            directory / f"{self._safe_key(partition_key)}.pkl"
            for directory in self.fallback_dirs
        ]
        for path in paths:
            if not path.exists():
                continue
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._hits += 1
                return data
            except Exception as exc:
                _logger.warning(
                    "[TfidfCache] Corrupt cache for %s at %s: %s",
                    partition_key,
                    path,
                    exc,
                )
        self._misses += 1
        return None

    def put(self, partition_key: str, artifacts: TfidfArtifacts) -> None:
        """Persist artifacts to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._key_path(partition_key)
        try:
            with open(path, "wb") as f:
                pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            _logger.warning("[TfidfCache] Failed to write cache for %s: %s", partition_key, exc)

    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        """Remove all cached artifacts for this config hash."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            _logger.info("[TfidfCache] Cleared %s", self.cache_dir)


__all__ = ["TfidfPersistentCache", "TfidfArtifacts"]
