"""Dense retrieval via FAISS for hybrid candidate selection.

Pre-computed embeddings per partition are stored as .npy files.
At query time: encode CRM name -> FAISS ANN lookup -> top-k indices.

Compatible with M4 Pro (CPU FAISS, ~1ms per lookup on 50k vectors).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-untyped]
except ImportError:
    faiss = None


def _faiss_available() -> bool:
    return faiss is not None


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
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.n, self.dim = embeddings.shape

        if self.n < 10_000:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            nlist = max(16, int(np.sqrt(self.n)))
            quantizer = faiss.IndexFlatIP(self.dim)
            
            # Product Quantization for extreme RAM reduction (compression 32x)
            # 384 dimensions -> 48 blocks of 8 bits each
            m = 48 if self.dim == 384 else (8 if self.dim == 64 else self.dim // 8)
            if self.dim % m != 0:
                m = 8 # Safe fallback
                
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT
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
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        if not _faiss_available():
            raise ImportError("faiss-cpu required")
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        obj.n = obj.index.ntotal
        obj.dim = obj.index.d
        return obj


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

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or Path(
            os.getenv("XGB_DENSE_STORE_DIR", "data/dense_index")
        )
        self._index_cache: OrderedDict[str, DenseIndex] = OrderedDict()
        self._max_cache = 20

    @staticmethod
    def _safe_key(partition_key: str) -> str:
        return partition_key.replace("|", "_").replace("/", "_").replace("\\", "_")

    def _embeddings_path(self, partition_key: str) -> Path:
        return self.store_dir / f"{self._safe_key(partition_key)}_embeddings.npy"

    def _index_path(self, partition_key: str) -> Path:
        return self.store_dir / f"{self._safe_key(partition_key)}_faiss.index"

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

    def get_index(self, partition_key: str) -> Optional[DenseIndex]:
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
                idx = DenseIndex.load(idx_path)
                self._put_cache(partition_key, idx)
                return idx
            except Exception as exc:
                _logger.warning("[DenseStore] Corrupt index %s: %s", idx_path, exc)

        emb = self.load_embeddings(partition_key)
        if emb is None:
            return None

        idx = DenseIndex(emb)
        idx.save(idx_path)
        self._put_cache(partition_key, idx)
        return idx

    def _put_cache(self, key: str, idx: DenseIndex) -> None:
        self._index_cache[key] = idx
        while len(self._index_cache) > self._max_cache:
            self._index_cache.popitem(last=False)


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
    "encode_query",
]
