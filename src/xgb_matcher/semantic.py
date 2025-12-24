"""Semantic similarity utilities for name matching (optional).

Optimized for GPU batch encoding via MPS on Apple Silicon.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable, Optional, List, Tuple, Dict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ENCODERS: dict[str, "SentenceTransformer"] = {}

# Cache for batch-encoded embeddings (text -> normalized vector)
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


def _semantic_enabled() -> bool:
    return os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


def _model_name() -> str:
    return os.getenv("XGB_SEMANTIC_MODEL", _DEFAULT_MODEL)


def _device() -> Optional[str]:
    """Get device, default to MPS on Apple Silicon for batch ops."""
    device = os.getenv("XGB_SEMANTIC_DEVICE")
    if device:
        return device
    # Auto-detect MPS for batch operations
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def _batch_size() -> int:
    try:
        return int(os.getenv("XGB_SEMANTIC_BATCH_SIZE", "512"))
    except ValueError:
        return 512


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
            print(f"[Semantic] Loaded model '{model_name}' on device: {device}")
        else:
            encoder = SentenceTransformer(model_name)
            print(f"[Semantic] Loaded model '{model_name}' on device: cpu")
        _ENCODERS[model_name] = encoder
        return encoder
    except Exception as e:
        print(f"[Semantic] Failed to load model: {e}")
        return None


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _normalize_for_embedding(text: str) -> str:
    """Normalize text before embedding for better tokenization.
    
    Splits CamelCase, separates digits, normalizes whitespace.
    Example: 'DigitBoxing15Pro' -> 'Digit Boxing 15 Pro'
    """
    if not text:
        return ""
    # 1. Split CamelCase: 'DigitBoxing' -> 'Digit Boxing'
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # 2. Split acronyms from words: 'XMLParser' -> 'XML Parser'
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    # 3. Separate digits: 'Box2Ring' -> 'Box 2 Ring'
    text = re.sub(r'(\d+)', r' \1 ', text)
    # 4. Normalize whitespace
    return ' '.join(text.split())


# ──────────────────────────────────────────────────────────────────────────────
# BATCH ENCODING (NEW - GPU optimized)
# ──────────────────────────────────────────────────────────────────────────────

def batch_encode_texts(texts: List[str]) -> Dict[str, np.ndarray]:
    """Encode multiple texts in a single GPU batch call.
    
    Returns a dict mapping normalized text -> embedding vector.
    Uses cache to avoid re-encoding already seen texts.
    """
    global _EMBEDDING_CACHE
    
    if not _semantic_enabled():
        return {}
    
    # Normalize all texts
    normalized_texts = [_normalize_for_embedding(t) for t in texts]
    
    # Find texts not yet in cache
    texts_to_encode = []
    for norm_text in normalized_texts:
        if norm_text and norm_text not in _EMBEDDING_CACHE:
            texts_to_encode.append(norm_text)
    
    # Remove duplicates while preserving order
    texts_to_encode = list(dict.fromkeys(texts_to_encode))
    
    if texts_to_encode:
        encoder = _get_encoder(_model_name())
        if encoder is not None:
            # Batch encode on GPU
            embeddings = encoder.encode(
                texts_to_encode,
                batch_size=_batch_size(),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Already normalized
            )
            # Populate cache
            for text, emb in zip(texts_to_encode, embeddings):
                _EMBEDDING_CACHE[text] = emb.astype(np.float32, copy=False)
    
    # Return only the requested texts
    result = {}
    for orig, norm in zip(texts, normalized_texts):
        if norm and norm in _EMBEDDING_CACHE:
            result[orig] = _EMBEDDING_CACHE[norm]
    
    return result


def top2_semantic_similarities(
    crm_name: str,
    candidate_names: List[str],
) -> Tuple[float, float, float]:
    """Compute top-2 semantic similarities between CRM name and candidates.
    
    Returns: (max_sim, second_sim, gap)
    - max_sim: highest similarity score
    - second_sim: second highest similarity score
    - gap: max_sim - second_sim (semantic dominance)
    
    Uses batch encoding for efficiency.
    """
    if not crm_name or not candidate_names:
        return 0.0, 0.0, 0.0
    
    if not _semantic_enabled():
        return 0.0, 0.0, 0.0
    
    # Filter valid candidate names
    valid_names = [n for n in candidate_names if n]
    if not valid_names:
        return 0.0, 0.0, 0.0
    
    # Batch encode all texts at once
    all_texts = [crm_name] + valid_names
    embeddings = batch_encode_texts(all_texts)
    
    crm_emb = embeddings.get(crm_name)
    if crm_emb is None:
        return 0.0, 0.0, 0.0
    
    # Compute all similarities
    similarities = []
    for name in valid_names:
        cand_emb = embeddings.get(name)
        if cand_emb is not None:
            sim = float(np.dot(crm_emb, cand_emb))
            similarities.append(sim)
    
    if not similarities:
        return 0.0, 0.0, 0.0
    
    # Sort descending to get top-2
    similarities.sort(reverse=True)
    max_sim = similarities[0]
    second_sim = similarities[1] if len(similarities) > 1 else 0.0
    gap = max_sim - second_sim
    
    return max_sim, second_sim, gap


def get_cache_stats() -> Tuple[int, int]:
    """Return (cache_size, cache_memory_mb) for monitoring."""
    cache_size = len(_EMBEDDING_CACHE)
    if cache_size > 0:
        sample_emb = next(iter(_EMBEDDING_CACHE.values()))
        emb_size = sample_emb.nbytes
        memory_mb = (cache_size * emb_size) / (1024 * 1024)
    else:
        memory_mb = 0
    return cache_size, memory_mb


def clear_cache():
    """Clear the embedding cache to free memory."""
    global _EMBEDDING_CACHE
    _EMBEDDING_CACHE = {}


# ──────────────────────────────────────────────────────────────────────────────
# LEGACY FUNCTIONS (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1_000_000)
def _embed_cached(model_name: str, text: str) -> Optional[np.ndarray]:
    """Legacy cached embedding - prefer batch_encode_texts for new code."""
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
    """Embed text with preprocessing for better matching."""
    model_name = _model_name()
    normalized = _normalize_for_embedding(text)
    return _embed_cached(model_name, normalized)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def max_semantic_similarity(source_text: str, candidate_texts: Iterable[str]) -> float:
    """Legacy function - prefer top2_semantic_similarities for new code."""
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
    "batch_encode_texts",
    "top2_semantic_similarities",
    "get_cache_stats",
    "clear_cache",
]
