"""First-stage retrieval primitives, not a downstream pair classifier.

Documents are encoded offline. Only the query transform is learned, allowing
one index to serve all outer-fold models without rebuilding document vectors.
The exhaustive collector below is a correctness oracle for an indexed serving
implementation, not a proposed O(N) production implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

ALLOWED_FOLDS = frozenset((2, 3, 4))


def official_text(names: Sequence[str], address: str) -> str:
    """Caller must supply only names already officially bound to this site.

    No parent/sibling propagation or relation interpretation happens here.
    Identifiers, qualification, retrieval ranks and labels have no argument.
    """
    if isinstance(names, str) or not isinstance(address, str):
        raise ValueError('Expected separate official names and address')
    clean = list(dict.fromkeys(' '.join(n.split()) for n in names if n.strip()))
    text = f"noms: {' | '.join(clean)} adresse: {' '.join(address.split())}"
    if re.search(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)', text):
        raise ValueError('Identifier-shaped number in model text')
    return text


def unit_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Empty embedding')
    return values / norms


@dataclass(frozen=True)
class TrainingMembership:
    """Label-side grouping data; never passed to an encoder or similarity."""
    query_id: str
    fold: int
    positive_siren: str
    negative_siren: str


def oof_training_positions(rows: Sequence[TrainingMembership], heldout_fold: int,
                           heldout_sirens: set[str]) -> np.ndarray:
    if heldout_fold not in ALLOWED_FOLDS:
        raise ValueError('Forbidden heldout fold')
    if any(r.fold not in ALLOWED_FOLDS for r in rows):
        raise ValueError('Forbidden training fold')
    if len({r.query_id for r in rows}) != len(rows):
        raise ValueError('Duplicate training query')
    if any(not re.fullmatch(r'\d{9}', s) for r in rows
           for s in (r.positive_siren, r.negative_siren)):
        raise ValueError('Invalid grouping identity')
    # Exclude heldout entities both as positives AND as explicit negatives.
    return np.asarray([i for i, r in enumerate(rows)
                       if r.fold != heldout_fold
                       and r.positive_siren not in heldout_sirens
                       and r.negative_siren not in heldout_sirens], dtype=np.int64)


def fit_query_transform(queries: np.ndarray, positives: np.ndarray,
                        negatives: np.ndarray, memberships: Sequence[TrainingMembership],
                        heldout_fold: int, heldout_sirens: set[str],
                        ridge: float = 1., negative_weight: float = .25) -> tuple[np.ndarray, dict]:
    """Bounded closed-form baseline; not claimed equivalent to SC-Block.

    It learns a query-to-official-document metric, not an identity dictionary.
    Uncertain/operationally equivalent negatives must be excluded upstream.
    Both entity and outer-fold separation are mandatory before fitting.
    """
    if not np.isfinite([ridge, negative_weight]).all() or ridge <= 0 or not 0 <= negative_weight < 1:
        raise ValueError('Invalid regularization')
    q, p, n = map(unit_vectors, (queries, positives, negatives))
    if q.shape != p.shape or q.shape != n.shape or len(q) != len(memberships):
        raise ValueError('Training alignment mismatch')
    take = oof_training_positions(memberships, heldout_fold, heldout_sirens)
    if not len(take):
        raise ValueError('No safe OOF training rows')
    x = q[take].astype(np.float64)
    target = unit_vectors(p[take] - negative_weight * n[take]).astype(np.float64)
    identity = np.eye(q.shape[1])
    weight = np.linalg.solve(x.T @ x + ridge * identity, x.T @ target + ridge * identity)
    return weight.astype(np.float32), dict(
        heldout_fold=heldout_fold, training_folds=sorted({memberships[i].fold for i in take}),
        input_rows=len(q), fitted_rows=len(take), excluded_rows=len(q)-len(take),
        heldout_entity_overlap=0, identifiers_as_features=False,
        document_transform='IDENTITY', score='NORMALIZED_QUERY_TRANSFORM_DOT_DOCUMENT',
        ridge=ridge, negative_weight=negative_weight)


def transform_queries(queries: np.ndarray, weight: np.ndarray) -> np.ndarray:
    q = unit_vectors(queries)
    w = np.asarray(weight, dtype=np.float32)
    if w.shape != (q.shape[1], q.shape[1]) or not np.isfinite(w).all():
        raise ValueError('Invalid learned query transform')
    return unit_vectors(q @ w)


def exact_top100_reference(query: np.ndarray, documents: np.ndarray, siret_ids: Sequence[str],
                           limit: int = 100) -> list[tuple[str, float]]:
    """Search the ENTIRE supplied corpus; no input reservoir or pair rescoring."""
    if not 1 <= limit <= 100:
        raise ValueError('Invalid initial output cap')
    docs = unit_vectors(documents)
    q = unit_vectors(np.asarray(query).reshape(1, -1))
    if docs.shape[1] != q.shape[1] or len(docs) != len(siret_ids):
        raise ValueError('Index/vector alignment mismatch')
    if len(set(siret_ids)) != len(siret_ids) or any(not re.fullmatch(r'\d{14}', s) for s in siret_ids):
        raise ValueError('Index must contain unique official SIRET documents')
    scores = (docs @ q[0]).astype(np.float64)
    order = np.lexsort((np.asarray(siret_ids), -scores))[:limit]
    return [(siret_ids[i], float(scores[i])) for i in order]
