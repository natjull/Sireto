"""Persistent TF-IDF cache per partition.

Serializes (vectorizer, matrix, names, char_vec, char_mat, addr_vec, addr_mat)
to disk so they survive across runs.  Invalidated by RetrievalConfigV1 signature
hash — if retrieval params change, the cache is rebuilt automatically.
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile
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
        require_verified: bool = False,
    ):
        self.config_hash = str(config_hash)
        self.cache_dir = (cache_dir or _cache_root()) / config_hash
        self.fallback_dirs = [
            (cache_dir or _cache_root()) / item
            for item in (fallback_config_hashes or [])
            if item and item != config_hash
        ]
        self.require_verified = bool(require_verified)
        self._hits = 0
        self._misses = 0
        self._verification_rejections = 0

    @staticmethod
    def _safe_key(partition_key: str) -> str:
        return (
            partition_key.replace("|", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    def _key_path(self, partition_key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(partition_key)}.pkl"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".sha256.json")

    def _verified_path(
        self,
        path: Path,
        *,
        partition_key: str,
    ) -> bool:
        if not self.require_verified:
            return True
        sidecar_path = self._sidecar_path(path)
        try:
            record = json.loads(sidecar_path.read_text(encoding="utf-8"))
            valid = (
                record.get("schema_version")
                == "sireto-tfidf-cache-integrity-1"
                and record.get("config_hash") == self.config_hash
                and record.get("partition_key") == partition_key
                and int(record.get("size_bytes", -1)) == path.stat().st_size
                and record.get("sha256") == self._sha256(path)
            )
        except Exception:
            valid = False
        if not valid:
            self._verification_rejections += 1
        return valid

    def get(self, partition_key: str) -> Optional[TfidfArtifacts]:
        """Load cached artifacts. Returns None on miss or corruption."""
        paths = [self._key_path(partition_key)] + [
            directory / f"{self._safe_key(partition_key)}.pkl"
            for directory in self.fallback_dirs
        ]
        for path in paths:
            if not path.exists():
                continue
            if not self._verified_path(path, partition_key=partition_key):
                _logger.warning(
                    "[TfidfCache] Refusing unverified cache for %s at %s",
                    partition_key,
                    path,
                )
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
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.cache_dir,
                prefix=f".{path.name}.",
                delete=False,
            ) as f:
                pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
                temporary_path = Path(f.name)
            os.replace(temporary_path, path)
            if self.require_verified:
                record = {
                    "schema_version": "sireto-tfidf-cache-integrity-1",
                    "config_hash": self.config_hash,
                    "partition_key": partition_key,
                    "size_bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
                sidecar_path = self._sidecar_path(path)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    prefix=f".{sidecar_path.name}.",
                    delete=False,
                ) as handle:
                    json.dump(
                        record,
                        handle,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.write("\n")
                    temporary_sidecar = Path(handle.name)
                os.replace(temporary_sidecar, sidecar_path)
        except Exception as exc:
            _logger.warning("[TfidfCache] Failed to write cache for %s: %s", partition_key, exc)

    def stats(self) -> Dict[str, int]:
        output = {"hits": self._hits, "misses": self._misses}
        if self.require_verified:
            output["verification_rejections"] = self._verification_rejections
        return output

    def clear(self) -> None:
        """Remove all cached artifacts for this config hash."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            _logger.info("[TfidfCache] Cleared %s", self.cache_dir)


__all__ = ["TfidfPersistentCache", "TfidfArtifacts"]
