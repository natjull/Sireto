"""Deterministic, label-blind V4.12 unit retrieval.

This module deliberately does not import the historical retrieval, V4.1, or
dataset-builder modules.  It consumes the already certified V4.12 strict
stores and reproduces only the frozen sparse top-100 algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import socket
import sys
import time
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

try:  # Package import in tests and normal repository use.
    from . import v412_strict_stores as strict_stores
    from .v412_strict_stores import _build_aligned_pool
except ImportError:  # Private sandbox copy executed as a standalone script.
    import v412_strict_stores as strict_stores  # type: ignore[no-redef]
    from v412_strict_stores import _build_aligned_pool  # type: ignore[no-redef]


STOP = "STOP_V412_UNIT_RETRIEVAL"
WORKER_RUN_SPEC_SCHEMA = "sireto-v4.12-unit-retrieval-worker-run-spec-1"
WORKER_IDENTITY_SCHEMA = "sireto-v4.12-unit-retrieval-worker-build-identity-1"
INTEGRITY_SCHEMA = "sireto-v4.12-unit-retrieval-integrity-1"

CANDIDATE_CEILING = 100
SPARSE_TOP_K = 500
RRF_K = 60
CACHE_NAMESPACE = (
    "296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82"
)

QUERY_COLUMNS = (
    "query_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARTITION_KEY_RE = re.compile(r"^(?:[0-9]{5}_|_[0-9]{5})$")
_SIRET_RE = re.compile(r"^[0-9]{14}$")
_TFIDF_PUNCT_RE = re.compile(r"[^\w\s]")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d{1,6}\b")
_STREET_PREFIX_RE = re.compile(
    r"^\d+\s*(BIS|TER|QUATER|B|T|Q)?\s*",
    flags=re.IGNORECASE,
)

_STREET_STOPWORDS = {
    "RUE",
    "AV",
    "AVE",
    "AVENUE",
    "BD",
    "BOULEVARD",
    "CHE",
    "CHEMIN",
    "IMP",
    "IMPASSE",
    "ALL",
    "ALLEE",
    "PL",
    "PLACE",
    "SQ",
    "SQUARE",
    "ROUTE",
    "RTE",
    "QUAI",
    "SENTIER",
    "ZA",
    "ZI",
    "ZAC",
}
_LEGAL_STOPWORDS = {
    "SAS",
    "SASU",
    "SARL",
    "EURL",
    "SCI",
    "SCIC",
    "SCOP",
    "SA",
    "SNC",
    "SELARL",
    "SELAS",
    "SELASU",
}
_PERSON_UL_CODES = {
    "1000",
    "1100",
    "1200",
    "1300",
    "1400",
    "1500",
    "1600",
    "2110",
}
_ACCENT_REPLACEMENTS = {
    "É": "E",
    "È": "E",
    "Ê": "E",
    "Ë": "E",
    "à": "a",
    "â": "a",
    "ä": "a",
    "á": "a",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ë": "e",
    "î": "i",
    "ï": "i",
    "í": "i",
    "ô": "o",
    "ö": "o",
    "ó": "o",
    "û": "u",
    "ü": "u",
    "ú": "u",
    "ù": "u",
    "ç": "c",
    "À": "A",
    "Â": "A",
    "Ä": "A",
    "Á": "A",
    "Î": "I",
    "Ï": "I",
    "Í": "I",
    "Ô": "O",
    "Ö": "O",
    "Ó": "O",
    "Û": "U",
    "Ü": "U",
    "Ú": "U",
    "Ù": "U",
    "Ç": "C",
}

_RETRIEVAL_POLICY = {
    "candidate_ceiling": 100,
    "include_closed": False,
    "drop_unnamed": True,
    "tfidf_name_mode": "bag",
    "sparse_retrieval_enabled": True,
    "dense_retrieval_enabled": False,
    "prefilter_k": 500,
    "word_top_k": 500,
    "char_top_k_effective": 500,
    "address_top_k": 500,
    "name_fusion": "WORD_TOP500_CHAR_TOP500_MAX_THEN_SCORE_DESC_ROW_INDEX_ASC",
    "address_order": "RANK_SPARSE_SCORES_STABLE_ARGSORT_REVERSED",
    "prefilter_trigger_size": 1,
    "retrieval_budget": 100,
    "min_candidates": 50,
    "fusion_mode": "rrf",
    "sparse_channel_fusion_mode": "separate_rrf",
    "rrf_k": 60,
    "rescue_addr_hash": True,
    "rescue_numeric_tokens": True,
    "mega_insee_policy": "full_insee",
    "siren_siblings": False,
    "nonfinite_policy": "STOP",
    "lookup_missing_policy": "OMIT_AND_COUNT",
    "zero_or_one_pool_policy": "UNCHANGED_SCORE_ZERO_THEN_SIRET_ASC",
    "channels": ["sparse_name", "sparse_address", "rescue"],
    "intermediate_tie_break": [
        "rrf_score_desc",
        "best_channel_rank_asc",
        "str_row_index_asc",
    ],
    "final_tie_break": ["rrf_score_desc", "candidate_siret_asc"],
}
_TFIDF_POLICY = {
    "namespace": CACHE_NAMESPACE,
    "tfidf_config_artifact_hash": (
        "92b68d1f7aa386f181edbede280e58df72f8583d7663419d77da88300d241c61"
    ),
    "sparse_config_hash": (
        "aeaa671959fc00dcec2e8a5393976d1e68da9dfa5ae48ef4d836e9dbdc3c564e"
    ),
    "tuple_fields": [
        "name_vectorizer",
        "name_matrix",
        "names",
        "char_vectorizer",
        "char_matrix",
        "address_vectorizer",
        "address_matrix",
    ],
    "miss_policy": "STOP",
    "rebuild_allowed": False,
    "write_allowed": False,
}
_WORKER_DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
    "cache_rebuild_attempted": False,
    "positive_injection": False,
}
_WORKER_RUN_SPEC_KEYS = {
    "schema_version",
    "safe_input_build_id",
    "safe_runtime_manifest_sha256",
    "safe_queries_dev_sha256",
    "query_count",
    "query_id_payload_sha256",
    "routing_payload_sha256",
    "worker_policy_sha256",
    "worker_lock_projection_sha256",
    "parent_runner_sha256",
    "worker_source_hashes",
    "strict_stores_build_id",
    "strict_stores_manifest_sha256",
    "retrieval",
    "tfidf_cache",
    "runtime",
    "max_rss_bytes",
    "gate_a_run_spec",
    "declarations",
}


class UnitRetrievalError(RuntimeError):
    """Fail-closed unit retrieval error."""


def _stop(message: str) -> None:
    raise UnitRetrievalError(f"{STOP}: {message}")


@dataclass(frozen=True)
class UnitRetrievalResult:
    partition_key: str
    candidate_sirets: tuple[str, ...]
    raw_pool_count: int
    aligned_pool_count: int
    lookup_missing_count: int


def _normalize_code(value: object) -> str | None:
    if value is None:
        return None
    try:
        if str(value).strip() == "" or str(value).lower() == "nan":
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    return text


def route_query(query: Mapping[str, Any], partition_keys: Iterable[str]) -> str:
    """Route a safe CRM query to its frozen INSEE partition, then postcode."""

    if not isinstance(query, Mapping):
        _stop("query must be a mapping")
    keys = set(partition_keys)
    if any(type(key) is not str or _PARTITION_KEY_RE.fullmatch(key) is None for key in keys):
        _stop("invalid partition key inventory")
    insee = _normalize_code(query.get("crm_insee"))
    if insee is not None:
        key = f"{insee}_"
        if key in keys:
            return key
    postcode = _normalize_code(query.get("crm_postcode"))
    if postcode is not None:
        key = f"_{postcode}"
        if key in keys:
            return key
    _stop("no frozen partition for query")


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if text.strip() in ("[ND]", "[nd]", "ND", "nan", "NaN", "None"):
        return ""
    text = text.upper().replace("-", " ")
    for old, new in _ACCENT_REPLACEMENTS.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _normalize_name(value: Any, *, max_len: int = 100) -> str:
    base = _normalize_text(value)
    if not base:
        return ""
    tokens = [token for token in base.split() if token not in _LEGAL_STOPWORDS]
    cleaned = " ".join(tokens) or base
    if len(cleaned) <= max_len:
        return cleaned
    cutoff = cleaned[: max_len + 1]
    if " " in cutoff:
        cutoff = cutoff.rsplit(" ", 1)[0]
    return cutoff


def _normalize_text_for_tfidf(value: Any) -> str:
    base = _normalize_text(value)
    if not base:
        return ""
    base = _TFIDF_PUNCT_RE.sub(" ", base)
    tokens = " ".join(base.split()).split()
    acronyms: list[str] = []
    buffer: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            buffer.append(token)
        else:
            if len(buffer) >= 2:
                acronyms.append("".join(buffer))
            buffer = []
    if len(buffer) >= 2:
        acronyms.append("".join(buffer))
    tokens.extend(acronyms)
    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token:
            continue
        if token not in seen:
            output.append(token)
            seen.add(token)
        singular = None
        if token.isalpha() and len(token) >= 5:
            if token.endswith("AUX") and len(token) >= 6:
                singular = token[:-3] + "AL"
            elif token.endswith("S") and not token.endswith("SS"):
                singular = token[:-1]
            elif token.endswith("X"):
                singular = token[:-1]
        if singular and singular not in seen:
            output.append(singular)
            seen.add(singular)
    return " ".join(output)


def _extract_street_number(address: Any) -> str | None:
    if not address:
        return None
    match = re.match(r"^(\d+)", str(address).strip())
    return match.group(1) if match else None


def _extract_street_name(address: Any) -> str:
    if not address:
        return ""
    normalized = _normalize_text(address)
    return _STREET_PREFIX_RE.sub("", normalized).strip()


def _address_hash(street_number: Any, street_name: Any) -> str | None:
    if not street_name:
        return None
    number = str(street_number).strip() if street_number is not None else ""
    tokens = [
        token
        for token in _normalize_text(str(street_name)).split()
        if token and token not in _STREET_STOPWORDS
    ]
    street = " ".join(tokens)
    if not street:
        return None
    return f"{number}|{street}" if number else street


def _candidate_address_hash(candidate: Mapping[str, Any]) -> str | None:
    number = candidate.get("numeroVoie") or candidate.get("street_number")
    street_type = candidate.get("typeVoie") or candidate.get("street_type")
    street_name = candidate.get("libelleVoie") or candidate.get("street_name")
    street = " ".join(str(value) for value in (street_type, street_name) if value)
    return _address_hash(str(number) if number is not None else None, street)


def _valid_candidate_name(value: Any) -> str:
    normalized = _normalize_name(value)
    return normalized if normalized and not normalized.isdigit() and len(normalized) > 2 else ""


def _primary_name(candidate: Mapping[str, Any]) -> str:
    for field in (
        "enseigne1",
        "denomination",
        "enseigne2",
        "enseigne3",
        "sigle_ul",
        "denomination_usuelle_ul",
        "denomination_ul",
    ):
        if value := _valid_candidate_name(candidate.get(field)):
            return value
    pm_names = candidate.get("pm_dirigeant_names") or []
    if isinstance(pm_names, str):
        pm_names = [item.strip() for item in pm_names.split("|") if item.strip()]
    for item in pm_names:
        if value := _valid_candidate_name(item):
            return value
    if candidate.get("cj_ul") in _PERSON_UL_CODES and (
        candidate.get("prenom_usuel_ul") or candidate.get("nom_ul")
    ):
        person = " ".join(
            str(value)
            for value in (
                candidate.get("prenom_usuel_ul"),
                candidate.get("nom_ul"),
            )
            if value
        )
        return _valid_candidate_name(person)
    return ""


def _numeric_tokens(value: Any) -> set[str]:
    return set(_NUMERIC_TOKEN_RE.findall(str(value))) if value else set()


def _require_finite_sparse(value: Any, label: str) -> None:
    data = getattr(value, "data", None)
    if data is None:
        _stop(f"{label} is not sparse")
    try:
        if not np.isfinite(np.asarray(data)).all():
            _stop(f"{label} contains non-finite values")
    except TypeError:
        _stop(f"{label} contains non-numeric values")


def _sparse_product(vectorizer: Any, matrix: Any, query: str, label: str) -> Any:
    if vectorizer is None and matrix is None:
        return None
    if vectorizer is None or matrix is None or not sparse.issparse(matrix):
        _stop(f"{label} vectorizer/matrix pair is invalid")
    if not query:
        return None
    try:
        vector = vectorizer.transform([query])
    except Exception as exc:
        _stop(f"{label} query transform failed: {exc}")
    if not sparse.issparse(vector):
        _stop(f"{label} query transform is not sparse")
    _require_finite_sparse(vector, f"{label} query")
    try:
        row = (vector @ matrix.T).getrow(0)
    except Exception as exc:
        _stop(f"{label} sparse product failed: {exc}")
    if not sparse.issparse(row):
        _stop(f"{label} product is not sparse")
    _require_finite_sparse(row, f"{label} product")
    return row


def _rank_sparse_scores(row: Any, top_k: int = SPARSE_TOP_K) -> list[tuple[int, float]]:
    if row is None or int(row.nnz) == 0:
        return []
    indices = row.indices
    values = row.data
    _require_finite_sparse(row, "sparse scores")
    if len(indices) > top_k:
        selected = np.argpartition(values, -top_k)[-top_k:]
        indices = indices[selected]
        values = values[selected]
    order = np.argsort(values, kind="stable")[::-1]
    return [(int(indices[index]), float(values[index])) for index in order]


def _rescue_indices(
    query: Mapping[str, Any],
    aligned_pool: Sequence[Mapping[str, Any]],
) -> list[int]:
    crm_name = query.get("crm_name") or ""
    crm_address = query.get("crm_address") or ""
    crm_hash = _address_hash(
        _extract_street_number(crm_address),
        _extract_street_name(crm_address),
    )
    crm_numbers = _numeric_tokens(crm_name)
    whitelisted: set[str] = set()
    index_by_siret: dict[str, int] = {}
    for index, candidate in enumerate(aligned_pool):
        siret = str(candidate.get("siret") or "")
        if not siret:
            continue
        index_by_siret[siret] = index
        address_match = bool(
            crm_hash and _candidate_address_hash(candidate) == crm_hash
        )
        numeric_match = bool(
            crm_numbers and (_numeric_tokens(_primary_name(candidate)) & crm_numbers)
        )
        if address_match or numeric_match:
            whitelisted.add(siret)
    return [
        index_by_siret[siret]
        for siret in sorted(whitelisted)
        if siret in index_by_siret
    ]


def _rrf(
    channels: Mapping[str, Sequence[int]],
    *,
    budget: int = CANDIDATE_CEILING,
    rrf_k: int = RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for channel_name, ordered_indices in channels.items():
        seen: set[int] = set()
        for rank, raw_index in enumerate(ordered_indices, start=1):
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
            ranks.setdefault(index, {})[channel_name] = rank
    ordered = sorted(
        scores,
        key=lambda index: (
            -scores[index],
            min(ranks[index].values()),
            str(index),
        ),
    )[:budget]
    return [(index, float(scores[index])) for index in ordered]


def _retrieve_internal(
    *,
    query: Mapping[str, Any],
    partition_store: Any,
    tfidf_cache: Any,
    lookup: Any,
) -> tuple[UnitRetrievalResult, int, int]:
    started = time.perf_counter_ns()
    partition_key = route_query(query, partition_store.partition_keys)
    rows = partition_store.load(partition_key)
    if not isinstance(rows, list):
        _stop("partition store did not return a list")
    raw_pool_count = len(rows)
    aligned_pool = _build_aligned_pool(rows)
    aligned_pool_count = len(aligned_pool)
    scores_by_index: dict[int, float] = {}

    if aligned_pool_count > 1:
        artifacts = tfidf_cache.get(partition_key, aligned_pool)
        if type(artifacts) is not tuple or len(artifacts) != 7:
            _stop("TF-IDF artifact tuple is invalid")
        (
            name_vectorizer,
            name_matrix,
            _names,
            char_vectorizer,
            char_matrix,
            address_vectorizer,
            address_matrix,
        ) = artifacts
        normalized_name = _normalize_text_for_tfidf(query.get("crm_name") or "")
        normalized_address = _normalize_text_for_tfidf(
            _normalize_text(query.get("crm_address") or "")
        )
        word_hits = _rank_sparse_scores(
            _sparse_product(
                name_vectorizer,
                name_matrix,
                normalized_name,
                "word name",
            )
        )
        char_hits = _rank_sparse_scores(
            _sparse_product(
                char_vectorizer,
                char_matrix,
                normalized_name,
                "character name",
            )
        )
        name_scores = dict(word_hits)
        for index, score in char_hits:
            name_scores[index] = max(name_scores.get(index, 0.0), score)
        name_hits = sorted(
            name_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:SPARSE_TOP_K]
        address_hits = _rank_sparse_scores(
            _sparse_product(
                address_vectorizer,
                address_matrix,
                normalized_address,
                "address",
            )
        )
        fused = _rrf(
            {
                "sparse_name": [index for index, _ in name_hits],
                "sparse_address": [index for index, _ in address_hits],
                "dense": [],
                "rescue": _rescue_indices(query, aligned_pool),
            }
        )
        selected_indices = [index for index, _ in fused]
        scores_by_index = dict(fused)
        selected = set(selected_indices)
        target = min(CANDIDATE_CEILING, aligned_pool_count)
        if len(selected_indices) < target:
            for index in range(aligned_pool_count):
                if index not in selected:
                    selected_indices.append(index)
                    scores_by_index[index] = 0.0
                if len(selected_indices) >= target:
                    break
    else:
        selected_indices = list(range(aligned_pool_count))
        scores_by_index = {index: 0.0 for index in selected_indices}

    selected_sirets: list[str] = []
    for index in selected_indices:
        if index < 0 or index >= aligned_pool_count:
            _stop("retrieval returned an out-of-range row index")
        siret = str(aligned_pool[index].get("siret") or "")
        if _SIRET_RE.fullmatch(siret) is None:
            _stop("retrieval returned a non-canonical SIRET")
        selected_sirets.append(siret)
    if len(selected_sirets) > CANDIDATE_CEILING:
        _stop("candidate ceiling exceeded before lookup")
    if len(set(selected_sirets)) != len(selected_sirets):
        _stop("duplicate SIRET before lookup")
    retrieval_ns = time.perf_counter_ns() - started

    lookup_started = time.perf_counter_ns()
    details = lookup.get_candidate_scene_details(selected_sirets)
    if not isinstance(details, Mapping):
        _stop("lookup did not return a mapping")
    unexpected = set(details) - set(selected_sirets)
    if unexpected:
        _stop("lookup returned an unrequested SIRET")
    missing_count = len(set(selected_sirets) - set(details))
    index_by_siret = {
        str(aligned_pool[index].get("siret") or ""): index
        for index in selected_indices
    }
    active: list[tuple[float, str]] = []
    for siret in selected_sirets:
        detail = details.get(siret)
        if detail is None:
            continue
        if not isinstance(detail, Mapping):
            _stop("lookup detail is not a mapping")
        if detail.get("candidate_state") == "A":
            active.append((scores_by_index[index_by_siret[siret]], siret))
    active.sort(key=lambda item: (-item[0], item[1]))
    final_sirets = tuple(siret for _, siret in active[:CANDIDATE_CEILING])
    if len(set(final_sirets)) != len(final_sirets):
        _stop("duplicate SIRET after lookup")
    lookup_ns = time.perf_counter_ns() - lookup_started
    return (
        UnitRetrievalResult(
            partition_key=partition_key,
            candidate_sirets=final_sirets,
            raw_pool_count=raw_pool_count,
            aligned_pool_count=aligned_pool_count,
            lookup_missing_count=missing_count,
        ),
        retrieval_ns,
        lookup_ns,
    )


def retrieve_unit_query(
    *,
    query: Mapping[str, Any],
    partition_store: Any,
    tfidf_cache: Any,
    lookup: Any,
) -> UnitRetrievalResult:
    """Retrieve at most 100 active SIRETs for one safe CRM query."""

    result, _retrieval_ns, _lookup_ns = _retrieve_internal(
        query=query,
        partition_store=partition_store,
        tfidf_cache=tfidf_cache,
        lookup=lookup,
    )
    return result


def _strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                _stop(f"{label} contains duplicate JSON keys")
            output[key] = value
        return output

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except UnitRetrievalError:
        raise
    except Exception as exc:
        _stop(f"{label} JSON parse failed: {exc}")
    if type(value) is not dict:
        _stop(f"{label} must be a JSON object")
    return value


def _read_inherited_fd(path: str, label: str) -> bytes:
    match = re.fullmatch(r"/dev/fd/([0-9]+)", path)
    if match is None:
        _stop(f"{label} must be passed through /dev/fd/N")
    descriptor = int(match.group(1))
    try:
        duplicate = os.dup(descriptor)
        before = os.fstat(duplicate)
        chunks: list[bytes] = []
        while block := os.read(duplicate, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(duplicate)
    except Exception as exc:
        _stop(f"{label} FD read failed: {exc}")
    finally:
        if "duplicate" in locals():
            os.close(duplicate)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        _stop(f"{label} FD changed while read")
    return b"".join(chunks)


def _validate_sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _stop(f"{label} is not a SHA-256")
    return value


def _runtime_values() -> dict[str, str]:
    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            _stop(f"runtime package missing: {distribution}")

    import duckdb
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pa.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": version("joblib"),
        "duckdb": duckdb.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _validate_worker_run_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _WORKER_RUN_SPEC_KEYS:
        _stop("worker run-spec keyset changed")
    if value["schema_version"] != WORKER_RUN_SPEC_SCHEMA:
        _stop("worker run-spec schema changed")
    for field in (
        "safe_input_build_id",
        "safe_runtime_manifest_sha256",
        "safe_queries_dev_sha256",
        "query_id_payload_sha256",
        "routing_payload_sha256",
        "worker_policy_sha256",
        "worker_lock_projection_sha256",
        "parent_runner_sha256",
        "strict_stores_build_id",
        "strict_stores_manifest_sha256",
    ):
        _validate_sha(value[field], field)
    if type(value["query_count"]) is not int or value["query_count"] < 0:
        _stop("invalid query count")
    if type(value["max_rss_bytes"]) is not int or value["max_rss_bytes"] <= 0:
        _stop("invalid RSS limit")
    if value["retrieval"] != _RETRIEVAL_POLICY:
        _stop("retrieval policy changed")
    if value["tfidf_cache"] != _TFIDF_POLICY:
        _stop("TF-IDF policy changed")
    if value["declarations"] != _WORKER_DECLARATIONS:
        _stop("worker declarations changed")
    if type(value["worker_source_hashes"]) is not dict:
        _stop("worker source hashes must be an object")
    for name, digest in value["worker_source_hashes"].items():
        if type(name) is not str:
            _stop("worker source name is invalid")
        _validate_sha(digest, f"worker source {name}")
    runtime = _runtime_values()
    if value["runtime"] != runtime:
        _stop("runtime differs from worker run-spec")
    try:
        gate = strict_stores._validate_run_spec(value["gate_a_run_spec"])
    except Exception as exc:
        _stop(f"encapsulated Gate A run-spec is invalid: {exc}")
    if gate["safe_input_build_id"] != value["safe_input_build_id"]:
        _stop("safe input build differs from Gate A")
    if gate["query_count"] != value["query_count"]:
        _stop("query count differs from Gate A")
    if gate["routing_payload_sha256"] != value["routing_payload_sha256"]:
        _stop("routing payload differs from Gate A")
    if gate["max_rss_bytes"] != value["max_rss_bytes"]:
        _stop("RSS limit differs from Gate A")
    output = dict(value)
    output["gate_a_run_spec"] = gate
    return output


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _worker_identity(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": WORKER_IDENTITY_SCHEMA,
        "worker_policy_sha256": run_spec["worker_policy_sha256"],
        "worker_lock_projection_sha256": run_spec[
            "worker_lock_projection_sha256"
        ],
        "parent_runner_sha256": run_spec["parent_runner_sha256"],
        "worker_source_hashes": run_spec["worker_source_hashes"],
        "safe_input_build_id": run_spec["safe_input_build_id"],
        "safe_runtime_manifest_sha256": run_spec[
            "safe_runtime_manifest_sha256"
        ],
        "safe_queries_dev_sha256": run_spec["safe_queries_dev_sha256"],
        "strict_stores_build_id": run_spec["strict_stores_build_id"],
        "strict_stores_manifest_sha256": run_spec[
            "strict_stores_manifest_sha256"
        ],
        "retrieval": run_spec["retrieval"],
        "tfidf_cache": run_spec["tfidf_cache"],
        "runtime": run_spec["runtime"],
    }


def compute_worker_build_id(run_spec: Mapping[str, Any]) -> str:
    """Compute the contractually isolated worker identity."""

    return hashlib.sha256(_canonical_json(_worker_identity(run_spec))).hexdigest()


def _query_table(data: bytes, run_spec: Mapping[str, Any]) -> pa.Table:
    if hashlib.sha256(data).hexdigest() != run_spec["safe_queries_dev_sha256"]:
        _stop("safe query Parquet hash mismatch")
    try:
        table = pq.read_table(pa.BufferReader(data), columns=list(QUERY_COLUMNS))
    except Exception as exc:
        _stop(f"safe query Parquet read failed: {exc}")
    expected_schema = pa.schema(
        [pa.field(name, pa.string(), nullable=False) for name in QUERY_COLUMNS]
    )
    if table.schema.metadata is not None or table.schema != expected_schema:
        _stop("safe query schema or metadata changed")
    if table.num_rows != run_spec["query_count"]:
        _stop("safe query count changed")
    rows = table.to_pylist()
    query_ids = [row["query_id"] for row in rows]
    if len(set(query_ids)) != len(query_ids):
        _stop("safe query IDs are not unique")
    expected_order = sorted(
        query_ids,
        key=lambda value: (
            hashlib.sha256(("v412-unit-engine:" + value).encode()).hexdigest(),
            value,
        ),
    )
    if query_ids != expected_order:
        _stop("safe query order changed")
    payload = "".join(f"{query_id}\n" for query_id in query_ids).encode()
    if hashlib.sha256(payload).hexdigest() != run_spec["query_id_payload_sha256"]:
        _stop("safe query ID payload changed")
    return table


def _require_denied_open(path: str, label: str) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except PermissionError as exc:
        if exc.errno == errno.EPERM:
            return True
        _stop(f"{label} denial returned errno={exc.errno}")
    except OSError as exc:
        _stop(f"{label} denial returned errno={exc.errno}")
    else:
        os.close(descriptor)
        _stop(f"{label} was readable")


def _require_network_denied() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.1)
        sock.connect(("127.0.0.1", 9))
    except PermissionError as exc:
        if exc.errno == errno.EPERM:
            return True
        _stop(f"network denial returned errno={exc.errno}")
    except OSError as exc:
        _stop(f"network denial returned errno={exc.errno}")
    finally:
        sock.close()
    _stop("network access was not denied")


def _require_write_denied(path: Path) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except PermissionError as exc:
        if exc.errno == errno.EPERM:
            return True
        _stop(f"write denial returned errno={exc.errno}")
    except OSError as exc:
        _stop(f"write denial returned errno={exc.errno}")
    else:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        _stop("write outside staging was not denied")


def _peak_rss_bytes() -> int:
    # Darwin reports bytes; Linux reports KiB.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _ensure_output_dir(path: str) -> Path:
    if path != "output":
        _stop("output directory must be the canonical relative path 'output'")
    root = Path.cwd()
    output = root / path
    if output.is_symlink() or not output.is_dir():
        _stop("output directory is absent or unsafe")
    if output.resolve() != (root.resolve() / "output"):
        _stop("output directory escaped the private run root")
    if any(output.iterdir()):
        _stop("output directory is not empty")
    return output


def _write_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table.replace_schema_metadata(None), temporary, compression="zstd")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_child_worker(
    *,
    run_spec_path: str,
    lookup_descriptor_path: str,
    queries_path: str,
    output_dir: str,
    worker_build_id: str,
    forbidden_oracle: str,
    forbidden_oracle_audit: str,
    forbidden_historical: str,
    forbidden_model: str,
) -> dict[str, Any]:
    """Execute the sandboxed child worker and write its three exact outputs."""

    total_started = time.perf_counter_ns()
    _validate_sha(worker_build_id, "worker build ID")
    run_spec_bytes = _read_inherited_fd(run_spec_path, "worker run-spec")
    descriptor_bytes = _read_inherited_fd(
        lookup_descriptor_path,
        "lookup descriptor",
    )
    queries_bytes = _read_inherited_fd(queries_path, "safe queries")
    run_spec = _validate_worker_run_spec(
        _strict_json_bytes(run_spec_bytes, "worker run-spec")
    )
    if compute_worker_build_id(run_spec) != worker_build_id:
        _stop("worker build ID mismatch")
    gate = run_spec["gate_a_run_spec"]
    if hashlib.sha256(descriptor_bytes).hexdigest() != gate[
        "lookup_descriptor_sha256"
    ]:
        _stop("lookup descriptor hash mismatch")
    descriptor = _strict_json_bytes(descriptor_bytes, "lookup descriptor")
    queries = _query_table(queries_bytes, run_spec)
    output = _ensure_output_dir(output_dir)

    sandbox_checks = {
        "allowed_read": True,
        "oracle_denied": _require_denied_open(forbidden_oracle, "oracle"),
        "oracle_audit_denied": _require_denied_open(
            forbidden_oracle_audit,
            "oracle audit",
        ),
        "historical_denied": _require_denied_open(
            forbidden_historical,
            "historical candidates",
        ),
        "model_denied": _require_denied_open(forbidden_model, "model"),
        "network_denied": _require_network_denied(),
        "write_denied": _require_write_denied(
            Path.cwd() / "write-denied-sentinel"
        ),
    }

    allowed = gate["allowed_read_files"]
    partition_store = strict_stores.StrictPartitionStore(
        gate["partition_records"],
        allowed,
        max_cache_entries=5,
    )
    tfidf_cache = strict_stores.StrictVerifiedTfidfCache(
        gate["cache_records"],
        allowed,
        namespace=CACHE_NAMESPACE,
        max_cache_entries=20,
    )
    if partition_store.partition_keys != tfidf_cache.partition_keys:
        _stop("partition/cache key sets differ")

    query_rows = queries.to_pylist()
    status_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    routing_payload = bytearray()
    retrieval_ns = 0
    lookup_ns = 0
    lookup_missing_count = 0
    with strict_stores.StrictSnapshotLookup(descriptor, allowed) as lookup:
        for query in query_rows:
            result, query_retrieval_ns, query_lookup_ns = _retrieve_internal(
                query=query,
                partition_store=partition_store,
                tfidf_cache=tfidf_cache,
                lookup=lookup,
            )
            retrieval_ns += query_retrieval_ns
            lookup_ns += query_lookup_ns
            lookup_missing_count += result.lookup_missing_count
            query_id = str(query["query_id"])
            routing_payload.extend(
                query_id.encode("utf-8")
                + b"\0"
                + result.partition_key.encode("utf-8")
                + b"\n"
            )
            status_rows.append(
                {
                    "query_id": query_id,
                    "candidate_count": len(result.candidate_sirets),
                }
            )
            candidate_rows.extend(
                {
                    "query_id": query_id,
                    "candidate_rank": rank,
                    "candidate_siret": siret,
                }
                for rank, siret in enumerate(result.candidate_sirets, start=1)
            )
            if _peak_rss_bytes() > run_spec["max_rss_bytes"]:
                _stop("RSS limit exceeded")
    if hashlib.sha256(routing_payload).hexdigest() != run_spec[
        "routing_payload_sha256"
    ]:
        _stop("routing payload changed")

    serialization_started = time.perf_counter_ns()
    status_schema = pa.schema(
        [
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("candidate_count", pa.uint8(), nullable=False),
        ]
    )
    candidate_schema = pa.schema(
        [
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("candidate_rank", pa.uint8(), nullable=False),
            pa.field("candidate_siret", pa.string(), nullable=False),
        ]
    )
    status_table = pa.Table.from_pylist(status_rows, schema=status_schema)
    candidate_table = pa.Table.from_pylist(candidate_rows, schema=candidate_schema)
    _write_parquet(output / "query_status.parquet", status_table)
    _write_parquet(output / "candidates_top100.parquet", candidate_table)

    candidate_payload = bytearray()
    for row in candidate_rows:
        candidate_payload.extend(
            row["query_id"].encode("utf-8")
            + b"\0"
            + row["candidate_siret"].encode("ascii")
            + b"\0"
            + str(row["candidate_rank"]).encode("ascii")
            + b"\n"
        )
    status_payload = bytearray()
    for row in status_rows:
        status_payload.extend(
            row["query_id"].encode("utf-8")
            + b"\0"
            + str(row["candidate_count"]).encode("ascii")
            + b"\n"
        )
    counts = [row["candidate_count"] for row in status_rows]
    serialization_ns = time.perf_counter_ns() - serialization_started
    total_ns = time.perf_counter_ns() - total_started
    peak_rss = _peak_rss_bytes()
    if peak_rss > run_spec["max_rss_bytes"]:
        _stop("RSS limit exceeded")
    integrity = {
        "schema_version": INTEGRITY_SCHEMA,
        "worker_build_id": worker_build_id,
        "query_count": len(status_rows),
        "candidate_count": len(candidate_rows),
        "minimum_pool_size": min(counts) if counts else 0,
        "maximum_pool_size": max(counts) if counts else 0,
        "under_ceiling_query_count": sum(
            count < CANDIDATE_CEILING for count in counts
        ),
        "empty_query_count": sum(count == 0 for count in counts),
        "lookup_missing_count": lookup_missing_count,
        "candidate_payload_bytes": len(candidate_payload),
        "candidate_payload_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "status_payload_bytes": len(status_payload),
        "status_payload_sha256": hashlib.sha256(status_payload).hexdigest(),
        "sandbox_checks": sandbox_checks,
        "peak_rss_bytes": peak_rss,
        "durations_ns": {
            "retrieval": retrieval_ns,
            "lookup": lookup_ns,
            "serialization": serialization_ns,
            "total": total_ns,
        },
        "declarations": dict(_WORKER_DECLARATIONS),
    }
    _write_json(output / "integrity.json", integrity)
    directory_fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return integrity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V4.12 unit retrieval worker")
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--lookup-descriptor", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-build-id", required=True)
    parser.add_argument("--forbidden-oracle", required=True)
    parser.add_argument("--forbidden-oracle-audit", required=True)
    parser.add_argument("--forbidden-historical", required=True)
    parser.add_argument("--forbidden-model", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        integrity = run_child_worker(
            run_spec_path=args.run_spec,
            lookup_descriptor_path=args.lookup_descriptor,
            queries_path=args.queries,
            output_dir=args.output_dir,
            worker_build_id=args.worker_build_id,
            forbidden_oracle=args.forbidden_oracle,
            forbidden_oracle_audit=args.forbidden_oracle_audit,
            forbidden_historical=args.forbidden_historical,
            forbidden_model=args.forbidden_model,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "verdict": "SEALED_V412_UNIT_RETRIEVAL",
                "worker_build_id": integrity["worker_build_id"],
                "query_count": integrity["query_count"],
                "candidate_count": integrity["candidate_count"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
