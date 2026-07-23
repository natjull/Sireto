"""Dense retrieval via FAISS for hybrid candidate selection.

Pre-computed embeddings per partition are stored as .npy files.
At query time: encode CRM name -> FAISS ANN lookup -> top-k indices.

Compatible with M4 Pro (CPU FAISS, ~1ms per lookup on 50k vectors).
"""

from __future__ import annotations

import logging
import os
import json
import hashlib
from importlib.util import find_spec
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)
faiss = None


def _get_faiss():
    global faiss
    if faiss is None:
        import faiss as faiss_module  # type: ignore[import-untyped]

        faiss = faiss_module
    return faiss


def _faiss_available() -> bool:
    return find_spec("faiss") is not None


# ---------------------------------------------------------------------------
# Dense Index wrapper
# ---------------------------------------------------------------------------


class DenseIndex:
    """FAISS-backed dense index for a single partition.

    Uses IndexFlatIP (brute-force inner product) for small partitions (<10k)
    and IndexIVFFlat for larger ones.  All vectors must be L2-normalized
    so that inner product == cosine similarity.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        if not _faiss_available():
            raise ImportError(
                "faiss-cpu required for dense retrieval. "
                "Install with: pip install faiss-cpu"
            )
        faiss_module = _get_faiss()
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.n, self.dim = embeddings.shape

        if self.n < 10_000:
            self.index = faiss_module.IndexFlatIP(self.dim)
        else:
            nlist = max(16, int(np.sqrt(self.n)))
            quantizer = faiss_module.IndexFlatIP(self.dim)
            
            # Product Quantization for extreme RAM reduction (compression 32x)
            # 384 dimensions -> 48 blocks of 8 bits each
            m = 48 if self.dim == 384 else (8 if self.dim == 64 else self.dim // 8)
            if self.dim % m != 0:
                m = 8 # Safe fallback
                
            self.index = faiss_module.IndexIVFPQ(
                quantizer,
                self.dim,
                nlist,
                m,
                8,
                faiss_module.METRIC_INNER_PRODUCT,
            )
            self.index.nprobe = min(16, nlist)
            self.index.train(embeddings)

        self.index.add(embeddings)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) for top-k nearest neighbours."""
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        k = min(k, self.n)
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]

    def save(self, path: Path) -> None:
        _get_faiss().write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        if not _faiss_available():
            raise ImportError("faiss-cpu required")
        faiss_module = _get_faiss()
        obj = cls.__new__(cls)
        obj.index = faiss_module.read_index(str(path))
        obj.n = obj.index.ntotal
        obj.dim = obj.index.d
        return obj


class _RemoteDenseIndex:
    """Thin index proxy backed by a clean persistent FAISS process."""

    def __init__(self, path: Path, client: Any) -> None:
        self.path = path
        self._client = client
        self.n, self.dim = client.describe(path)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._client.search(self.path, query, min(k, self.n))


# ---------------------------------------------------------------------------
# Partition Embedding Store
# ---------------------------------------------------------------------------


class PartitionEmbeddingStore:
    """Manages pre-computed embeddings and FAISS indices per partition.

    Directory layout:
        <store_dir>/
            <partition_key>_embeddings.npy   (N x dim, float32)
            <partition_key>_faiss.index      (serialized FAISS)
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        *,
        expected_model_fingerprint: str | None = None,
    ) -> None:
        self.store_dir = store_dir or Path(
            os.getenv("XGB_DENSE_STORE_DIR", "data/dense_index")
        )
        self._index_cache: OrderedDict[str, _RemoteDenseIndex] = OrderedDict()
        self._max_cache = 20
        self._faiss_client = None
        self.expected_model_fingerprint = expected_model_fingerprint

    @staticmethod
    def _safe_key(partition_key: str) -> str:
        return partition_key.replace("|", "_").replace("/", "_").replace("\\", "_")

    def _embeddings_path(self, partition_key: str) -> Path:
        return self.store_dir / f"{self._safe_key(partition_key)}_embeddings.npy"

    def _index_path(self, partition_key: str) -> Path:
        return self.store_dir / f"{self._safe_key(partition_key)}_faiss.index"

    def _manifest_path(self, partition_key: str) -> Path:
        return self.store_dir / f"{self._safe_key(partition_key)}_manifest.json"

    # -- Read / Write embeddings --

    def has_embeddings(self, partition_key: str) -> bool:
        return self._index_path(partition_key).exists()

    def save_embeddings(self, partition_key: str, embeddings: np.ndarray) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._embeddings_path(partition_key), embeddings.astype(np.float32))

    def load_embeddings(self, partition_key: str) -> Optional[np.ndarray]:
        path = self._embeddings_path(partition_key)
        if not path.exists():
            return None
        return np.load(path).astype(np.float32)

    # -- FAISS index management --

    def _client(self):
        if self._faiss_client is None:
            from .faiss_process import FaissSearchProcessClient

            self._faiss_client = FaissSearchProcessClient(
                max_cache=self._max_cache
            )
        return self._faiss_client

    def get_index(self, partition_key: str) -> Optional[_RemoteDenseIndex]:
        """Load or build a FAISS index for the partition.

        Priority:
        1. In-memory cache
        2. On-disk FAISS index
        3. Build from .npy embeddings (and persist the index)
        """
        if not _faiss_available():
            return None

        if partition_key in self._index_cache:
            self._index_cache.move_to_end(partition_key)
            return self._index_cache[partition_key]

        idx_path = self._index_path(partition_key)
        if idx_path.exists():
            try:
                idx = _RemoteDenseIndex(idx_path, self._client())
                self._put_cache(partition_key, idx)
                return idx
            except Exception as exc:
                _logger.warning("[DenseStore] Corrupt index %s: %s", idx_path, exc)

        emb = self.load_embeddings(partition_key)
        if emb is None:
            return None

        from .faiss_process import build_faiss_index_isolated

        build_faiss_index_isolated(emb, idx_path)
        idx = _RemoteDenseIndex(idx_path, self._client())
        self._put_cache(partition_key, idx)
        return idx

    def validates_candidate_order(
        self,
        partition_key: str,
        candidates: List[dict],
        index: _RemoteDenseIndex,
    ) -> bool:
        """Reject an ANN index whose row ids cannot map to this exact pool."""
        if index.n != len(candidates):
            _logger.error(
                "[DenseStore] Cardinality mismatch for %s: index=%d pool=%d",
                partition_key,
                index.n,
                len(candidates),
            )
            return False
        manifest_path = self._manifest_path(partition_key)
        if not manifest_path.exists():
            _logger.warning(
                "[DenseStore] Legacy index without order manifest: %s",
                partition_key,
            )
            return True
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _logger.error(
                "[DenseStore] Invalid manifest for %s: %s",
                partition_key,
                error,
            )
            return False
        payload = "\n".join(
            str(candidate.get("siret") or "") for candidate in candidates
        )
        observed_hash = hashlib.sha256(
            payload.encode("ascii", errors="ignore")
        ).hexdigest()
        aligned = (
            int(manifest.get("candidate_count", -1)) == len(candidates)
            and manifest.get("siret_order_sha256") == observed_hash
        )
        if not aligned:
            _logger.error(
                "[DenseStore] Candidate order mismatch for %s",
                partition_key,
            )
            return False
        if (
            self.expected_model_fingerprint is not None
            and manifest.get("semantic_model_fingerprint")
            != self.expected_model_fingerprint
        ):
            _logger.error(
                "[DenseStore] Semantic model mismatch for %s",
                partition_key,
            )
            return False
        return True

    def _put_cache(self, key: str, idx: _RemoteDenseIndex) -> None:
        self._index_cache[key] = idx
        while len(self._index_cache) > self._max_cache:
            self._index_cache.popitem(last=False)

    def close(self) -> None:
        if self._faiss_client is not None:
            self._faiss_client.close()
            self._faiss_client = None
        self._index_cache.clear()


class GlobalDenseSirenIndex:
    """Read-only ANN index mapping a CRM name to global SIREN entities."""

    def __init__(
        self,
        index_dir: Path,
        *,
        expected_model_fingerprint: str | None = None,
    ) -> None:
        if not _faiss_available():
            raise ImportError("faiss-cpu required")
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        self.manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        if (
            expected_model_fingerprint is not None
            and self.manifest.get("semantic_model_fingerprint")
            != expected_model_fingerprint
        ):
            raise ValueError(
                "Global dense SIREN semantic model fingerprint mismatch"
            )
        from .faiss_process import FaissSearchProcessClient

        self._faiss_client = FaissSearchProcessClient(max_cache=1)
        self._index_path = self.index_dir / "siren_faiss.index"
        self.ntotal, self.dimension = self._faiss_client.describe(
            self._index_path
        )
        self.siren_ids = np.load(
            self.index_dir / "siren_ids.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if self.ntotal != len(self.siren_ids):
            raise ValueError("Global dense SIREN index/id cardinality mismatch")
        manifest_count = self.manifest.get("entity_count")
        if manifest_count is not None and int(manifest_count) != len(self.siren_ids):
            raise ValueError("Global dense SIREN manifest cardinality mismatch")

    def query(self, crm_name: str, top_k: int = 50) -> List[Tuple[str, float]]:
        query = encode_query(crm_name)
        if query is None or self.ntotal == 0:
            return []
        k = min(top_k, self.ntotal)
        scores, indices = self._faiss_client.search(
            self._index_path,
            query.astype(np.float32, copy=False),
            k,
        )
        hits: List[Tuple[str, float]] = []
        for index, score in zip(indices, scores, strict=True):
            if index < 0:
                continue
            raw_siren = self.siren_ids[int(index)]
            siren = (
                raw_siren.decode("ascii")
                if isinstance(raw_siren, bytes)
                else str(raw_siren)
            )
            hits.append((siren.zfill(9), float(score)))
        return hits

    def close(self) -> None:
        self._faiss_client.close()


# ---------------------------------------------------------------------------
# Query-time embedding (thin wrapper around semantic.py)
# ---------------------------------------------------------------------------


def encode_query(text: str) -> Optional[np.ndarray]:
    """Encode a single CRM query string to a dense vector.

    Uses the same model / preprocessing as the pre-computed partition embeddings.
    Returns None if semantic module is unavailable.
    """
    try:
        from .semantic import batch_encode_texts
    except ImportError:
        return None

    if not text:
        return None
    result = batch_encode_texts([text])
    vec = result.get(text)
    if vec is None:
        return None
    return vec.astype(np.float32)


__all__ = [
    "DenseIndex",
    "PartitionEmbeddingStore",
    "GlobalDenseSirenIndex",
    "encode_query",
]
