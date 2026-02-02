"""Centralized semantic embedding server for multiprocessing."""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np


def _batch_size() -> int:
    try:
        return int(os.getenv("XGB_SEMANTIC_BATCH_SIZE", "512"))
    except ValueError:
        return 512


def semantic_server_process(
    queue_in,
    responses,
    model_name: str,
    device: Optional[str] = None,
) -> None:
    """Process loop: receive text batches, return embeddings.

    Expects messages: (request_id, texts)
    Replies via responses[request_id] = np.ndarray
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sentence_transformers not installed") from exc

    if device:
        encoder = SentenceTransformer(model_name, device=device)
    else:
        encoder = SentenceTransformer(model_name)

    while True:
        item = queue_in.get()
        if item is None:
            break
        request_id, texts = item
        if not texts:
            responses[request_id] = np.zeros((0, 0), dtype=np.float32)
            continue
        try:
            embeddings = encoder.encode(
                texts,
                batch_size=_batch_size(),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            responses[request_id] = embeddings.astype(np.float32, copy=False)
        except Exception:
            responses[request_id] = None


class SemanticClient:
    """Client for the semantic server process."""

    def __init__(self, queue_in, responses, *, timeout_s: float = 120.0, poll_s: float = 0.01):
        self._queue_in = queue_in
        self._responses = responses
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._counter = 0
        self._pid = os.getpid()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        request_id = f"{self._pid}-{self._counter}"
        self._counter += 1
        self._queue_in.put((request_id, texts))
        start = time.monotonic()
        while True:
            if request_id in self._responses:
                embeddings = self._responses.pop(request_id)
                if embeddings is None:
                    return np.zeros((0, 0), dtype=np.float32)
                return embeddings
            if time.monotonic() - start > self._timeout_s:
                raise TimeoutError("Semantic server response timed out")
            time.sleep(self._poll_s)
