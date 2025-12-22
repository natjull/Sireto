"""Semantic similarity utilities for name matching (optional)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ENCODERS: dict[str, "SentenceTransformer"] = {}


def _semantic_enabled() -> bool:
    return os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


def _model_name() -> str:
    return os.getenv("XGB_SEMANTIC_MODEL", _DEFAULT_MODEL)


def _device() -> Optional[str]:
    device = os.getenv("XGB_SEMANTIC_DEVICE")
    return device or None


def _batch_size() -> int:
    try:
        return int(os.getenv("XGB_SEMANTIC_BATCH_SIZE", "64"))
    except ValueError:
        return 64


def _get_encoder(model_name: str) -> Optional["SentenceTransformer"]:
    if not _semantic_enabled():
        return None
    if SentenceTransformer is None:
        return None
    if model_name in _ENCODERS:
        return _ENCODERS[model_name]
    try:
        device = _device()
        if device:
            encoder = SentenceTransformer(model_name, device=device)
        else:
            encoder = SentenceTransformer(model_name)
        _ENCODERS[model_name] = encoder
        return encoder
    except Exception:
        return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


@lru_cache(maxsize=50000)
def _embed_cached(model_name: str, text: str) -> Optional[np.ndarray]:
    encoder = _get_encoder(model_name)
    if encoder is None:
        return None
    if not text:
        return None
    vec = encoder.encode(
        [text],
        batch_size=_batch_size(),
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return _normalize(vec.astype(np.float32, copy=False))


def embed_text(text: str) -> Optional[np.ndarray]:
    model_name = _model_name()
    return _embed_cached(model_name, text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def max_semantic_similarity(source_text: str, candidate_texts: Iterable[str]) -> float:
    if not source_text:
        return 0.0
    src_vec = embed_text(source_text)
    if src_vec is None:
        return 0.0
    best = 0.0
    for cand_text in candidate_texts:
        if not cand_text:
            continue
        cand_vec = embed_text(cand_text)
        if cand_vec is None:
            continue
        sim = float(np.dot(src_vec, cand_vec))
        if sim > best:
            best = sim
    return best


__all__ = [
    "embed_text",
    "cosine_similarity",
    "max_semantic_similarity",
]
