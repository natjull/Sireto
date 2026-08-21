"""Entity-disjoint contrastive rescue at the SIREN level.

The rescue learns a *similarity transform*, never an entity lookup.  CRM and
official texts are embedded by the same frozen multilingual encoder; two small
linear adapters are fitted with in-batch and explicit same-pool negatives.
SIREN/SIRET values are retained only for grouping, expansion and auditing and
are never included in model text or features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CONTRASTIVE_RESCUE_SCHEMA_VERSION = "sireto-contrastive-siren-rescue-v1"
_SPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return _SPACE.sub(" ", str(value or "").strip())


def query_text(row: Mapping[str, Any]) -> str:
    parts = [
        "nom:",
        normalize_text(row.get("crm_name")),
        "adresse:",
        normalize_text(row.get("crm_number")),
        normalize_text(row.get("crm_address")),
        normalize_text(row.get("crm_postcode")),
        normalize_text(row.get("crm_insee")),
    ]
    return normalize_text(" ".join(parts))


def candidate_text(row: Mapping[str, Any]) -> str:
    name_fields = (
        "denomination",
        "enseigne1",
        "enseigne2",
        "enseigne3",
        "sigle_ul",
        "denomination_ul",
        "denomination_usuelle_ul",
        "nom_ul",
        "prenom_usuel_ul",
        "pm_dirigeant_names",
    )
    names = list(dict.fromkeys(filter(None, (normalize_text(row.get(k)) for k in name_fields))))
    address = normalize_text(
        " ".join(
            filter(
                None,
                (
                    normalize_text(row.get("numeroVoie")),
                    normalize_text(row.get("typeVoie")),
                    normalize_text(row.get("libelleVoie")),
                    normalize_text(row.get("complementAdresse")),
                    normalize_text(row.get("postcode")),
                    normalize_text(row.get("city")),
                ),
            )
        )
    )
    return normalize_text(f"noms: {' | '.join(names[:12])} adresse: {address}")


def siren_documents(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate local SIRET records into bounded, deterministic SIREN docs."""
    if frame.empty:
        return pd.DataFrame(columns=["siren", "document_text"])
    working = frame.copy()
    working["siren"] = working["siren"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(9)
    working["_text"] = [candidate_text(row) for row in working.to_dict("records")]

    def aggregate(values: Iterable[str]) -> str:
        unique = list(dict.fromkeys(value for value in values if value))
        return " [site] ".join(unique[:8])

    return (
        working.groupby("siren", sort=True, as_index=False)["_text"]
        .agg(aggregate)
        .rename(columns={"_text": "document_text"})
    )


@dataclass(frozen=True)
class AdapterConfig:
    negative_weight: float = 0.25
    ridge: float = 1.0
    identity_regularization: float = 0.05
    seed: int = 20260821


def fit_linear_adapters(
    queries: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    config: AdapterConfig = AdapterConfig(),
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a stable contrastive query adapter by regularized least squares.

    The target direction is the positive document embedding minus a bounded
    component of the explicit hard negative.  This is the closed-form linear
    analogue of a triplet update; it avoids fine-tuning the multilingual
    encoder and keeps inference compatible with frozen document embeddings.
    """
    if queries.shape != positives.shape or queries.shape != negatives.shape:
        raise ValueError("query, positive and negative embeddings must share shape")
    if queries.ndim != 2 or not len(queries):
        raise ValueError("contrastive training requires non-empty 2-D embeddings")
    dimension = int(queries.shape[1])
    x = np.asarray(queries, dtype=np.float64)
    positive = np.asarray(positives, dtype=np.float64)
    negative = np.asarray(negatives, dtype=np.float64)
    target = positive - config.negative_weight * negative
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    identity = np.eye(dimension, dtype=np.float64)
    gram = x.T @ x + (config.ridge + config.identity_regularization) * identity
    cross = x.T @ target + config.identity_regularization * identity
    column_weight = np.linalg.solve(gram, cross)
    q_weight = column_weight.T.astype(np.float32)
    d_weight = np.eye(dimension, dtype=np.float32)

    before = apply_adapter(x.astype(np.float32), d_weight)
    after = apply_adapter(x.astype(np.float32), q_weight)
    positive32 = np.asarray(positives, dtype=np.float32)
    negative32 = np.asarray(negatives, dtype=np.float32)

    def loss(values: np.ndarray) -> float:
        positive_score = np.sum(values * positive32, axis=1)
        negative_score = np.sum(values * negative32, axis=1)
        return float(np.mean(np.maximum(0.0, 0.1 - positive_score + negative_score)))

    history = [
        {"stage": "identity", "triplet_margin_loss": loss(before)},
        {"stage": "fitted", "triplet_margin_loss": loss(after)},
    ]
    return q_weight, d_weight, history


def apply_adapter(embeddings: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def top_sirens(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    sirens: Sequence[str],
    limit: int,
) -> list[list[tuple[str, float]]]:
    if document_embeddings.shape[0] != len(sirens):
        raise ValueError("SIREN/document cardinality mismatch")
    if not len(document_embeddings):
        return [[] for _ in range(len(query_embeddings))]
    output: list[list[tuple[str, float]]] = []
    for query in query_embeddings:
        scores = document_embeddings @ query
        k = min(int(limit), len(scores))
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        output.append([(str(sirens[index]), float(scores[index])) for index in indices])
    return output


def rank_sites(frame: pd.DataFrame, row: Mapping[str, Any], siren: str, limit: int) -> pd.DataFrame:
    sites = frame[frame["siren"].astype(str).str.zfill(9) == str(siren)].copy()
    if sites.empty:
        return sites
    crm_number = normalize_text(row.get("crm_number"))
    crm_address = normalize_text(row.get("crm_address")).upper()
    sites["_number_match"] = sites.get("numeroVoie", "").fillna("").astype(str).map(normalize_text).eq(crm_number)
    street = (
        sites.get("typeVoie", "").fillna("").astype(str)
        + " "
        + sites.get("libelleVoie", "").fillna("").astype(str)
    ).map(normalize_text).str.upper()
    sites["_address_overlap"] = street.map(
        lambda value: len(set(value.split()) & set(crm_address.split()))
    )
    sites["_active"] = sites.get("etat_admin", "").fillna("").astype(str).str.upper().eq("A")
    sites["_siege"] = sites.get("is_siege", False).fillna(False).astype(bool)
    return sites.sort_values(
        ["_number_match", "_address_overlap", "_active", "_siege", "siret"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).head(limit)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_path: Path) -> str:
    payload: dict[str, str] = {}
    for name in ("config.json", "tokenizer_config.json", "modules.json"):
        path = model_path / name
        if path.is_file():
            payload[name] = sha256_file(path)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
