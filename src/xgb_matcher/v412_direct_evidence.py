"""Pure transformations and invariants for the V4.12 direct-evidence guard.

This module performs no I/O.  In particular, it cannot open labels, model
outputs, ranker pools, scenes, or challenge artifacts.
"""

from __future__ import annotations

import json
from numbers import Integral
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


QUERY_EVIDENCE_COLUMNS = [
    "query_id",
    "partition_key",
    "active_universe_count",
    "direct_candidate_count",
    "direct_siren_count",
    "sole_direct_siret",
    "sole_direct_siren",
    "cross_siren_direct_collision",
    "same_siren_direct_multisite",
    "evidence_refs_json",
]
CANDIDATE_EVIDENCE_COLUMNS = [
    "evidence_ref",
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "exact_name_anchor",
    "exact_address_anchor",
    "strong_name_evidence",
    "strong_address_evidence",
    "direct_evidence_class",
    "direct_match_rule",
]
GUARD_REASONS = {
    "NO_DIRECT_EVIDENCE",
    "MULTIPLE_STRONG_DIRECT_CANDIDATES",
    "DIRECT_EVIDENCE_DISAGREES_TOP1",
}
_SIRET = re.compile(r"^\d{14}$")
_SIREN = re.compile(r"^\d{9}$")


def evidence_ref(query_id: str, candidate_siret: str) -> str:
    """Return the stable, reversible reference stored in the query aggregate."""

    return f"DIRECT:{query_id}:{candidate_siret}"


def direct_match_rule(record: Mapping[str, Any]) -> str:
    raw_name = record.get("exact_name_anchor")
    raw_address = record.get("exact_address_anchor")
    if not isinstance(raw_name, (bool, np.bool_)) or not isinstance(
        raw_address, (bool, np.bool_)
    ):
        raise ValueError("STOP_V412_EVIDENCE: exact anchor is not boolean")
    exact_name = bool(raw_name)
    exact_address = bool(raw_address)
    if exact_name and exact_address:
        return "EXACT_NAME_AND_ADDRESS"
    if exact_name:
        return "EXACT_NAME_STRONG_ADDRESS"
    if exact_address:
        return "EXACT_ADDRESS_STRONG_NAME"
    raise ValueError("STOP_V412_EVIDENCE: direct candidate lacks exact anchor")


def candidate_evidence_record(
    query_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one result of the frozen V4 direct-evidence function."""

    if record.get("direct_evidence_class") != "NAME_AND_ADDRESS":
        raise ValueError("STOP_V412_EVIDENCE: non-direct candidate returned")
    return {
        "evidence_ref": evidence_ref(
            str(query_id), str(record.get("candidate_siret") or "")
        ),
        "query_id": str(query_id),
        "candidate_siret": str(record.get("candidate_siret") or ""),
        "candidate_siren": str(record.get("candidate_siren") or ""),
        "candidate_state": str(record.get("candidate_state") or ""),
        "exact_name_anchor": bool(record.get("exact_name_anchor")),
        "exact_address_anchor": bool(record.get("exact_address_anchor")),
        "strong_name_evidence": True,
        "strong_address_evidence": True,
        "direct_evidence_class": "NAME_AND_ADDRESS",
        "direct_match_rule": direct_match_rule(record),
    }


def query_evidence_record(
    *,
    query_id: str,
    partition_key: str,
    active_universe_count: int,
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = list(candidates)
    sirets = [str(row["candidate_siret"]) for row in candidates]
    sirens = [str(row["candidate_siren"]) for row in candidates]
    unique_sirens = set(sirens)
    sole = candidates[0] if len(candidates) == 1 else None
    return {
        "query_id": str(query_id),
        "partition_key": str(partition_key),
        "active_universe_count": int(active_universe_count),
        "direct_candidate_count": len(candidates),
        "direct_siren_count": len(unique_sirens),
        "sole_direct_siret": (
            str(sole["candidate_siret"]) if sole is not None else None
        ),
        "sole_direct_siren": (
            str(sole["candidate_siren"]) if sole is not None else None
        ),
        "cross_siren_direct_collision": len(unique_sirens) >= 2,
        "same_siren_direct_multisite": (
            len(candidates) >= 2 and len(unique_sirens) == 1
        ),
        "evidence_refs_json": json.dumps(
            sorted(
                evidence_ref(str(query_id), candidate_siret)
                for candidate_siret in sirets
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def build_evidence_frames(
    query_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_frame = pd.DataFrame(
        list(query_records), columns=QUERY_EVIDENCE_COLUMNS
    ).sort_values("query_id", kind="mergesort").reset_index(drop=True)
    candidate_frame = pd.DataFrame(
        list(candidate_records), columns=CANDIDATE_EVIDENCE_COLUMNS
    ).sort_values(
        ["query_id", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)
    validate_evidence(query_frame, candidate_frame)
    return query_frame, candidate_frame


def validate_evidence(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Fail closed on every cardinality, identity, state, and reference drift."""

    if list(queries.columns) != QUERY_EVIDENCE_COLUMNS:
        raise ValueError("STOP_V412_EVIDENCE: query evidence schema changed")
    if list(candidates.columns) != CANDIDATE_EVIDENCE_COLUMNS:
        raise ValueError("STOP_V412_EVIDENCE: candidate evidence schema changed")
    if queries["query_id"].astype(str).duplicated().any():
        raise ValueError("STOP_V412_EVIDENCE: duplicate query evidence")
    if queries["query_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("STOP_V412_EVIDENCE: empty query ID")
    if queries["partition_key"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("STOP_V412_EVIDENCE: empty partition key")
    for column in (
        "cross_siren_direct_collision",
        "same_siren_direct_multisite",
    ):
        if not queries[column].map(lambda value: type(value) is bool).all():
            raise ValueError(
                f"STOP_V412_EVIDENCE: invalid aggregate boolean {column}"
            )
    for column in (
        "active_universe_count",
        "direct_candidate_count",
        "direct_siren_count",
    ):
        if not queries[column].map(
            lambda value: (
                isinstance(value, Integral)
                and not isinstance(value, (bool, np.bool_))
                and int(value) >= 0
            )
        ).all():
            raise ValueError(
                f"STOP_V412_EVIDENCE: invalid aggregate count {column}"
            )
    if (
        queries["active_universe_count"].astype(int)
        < queries["direct_candidate_count"].astype(int)
    ).any():
        raise ValueError(
            "STOP_V412_EVIDENCE: direct count exceeds active universe"
        )
    if candidates.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("STOP_V412_EVIDENCE: duplicate candidate evidence")
    if candidates["query_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("STOP_V412_EVIDENCE: empty candidate query ID")
    if candidates["evidence_ref"].astype(str).duplicated().any():
        raise ValueError("STOP_V412_EVIDENCE: duplicate evidence reference")
    if not candidates["candidate_siret"].astype(str).map(_SIRET.fullmatch).all():
        raise ValueError("STOP_V412_EVIDENCE: invalid candidate SIRET")
    if not candidates["candidate_siren"].astype(str).map(_SIREN.fullmatch).all():
        raise ValueError("STOP_V412_EVIDENCE: invalid candidate SIREN")
    if (
        candidates["candidate_siret"].astype(str).str[:9]
        != candidates["candidate_siren"].astype(str)
    ).any():
        raise ValueError("STOP_V412_EVIDENCE: SIRET/SIREN mismatch")
    if candidates["candidate_state"].astype(str).ne("A").any():
        raise ValueError("STOP_V412_EVIDENCE: inactive candidate")
    boolean_columns = [
        "exact_name_anchor",
        "exact_address_anchor",
        "strong_name_evidence",
        "strong_address_evidence",
    ]
    for column in boolean_columns:
        if not candidates[column].map(lambda value: type(value) is bool).all():
            raise ValueError(f"STOP_V412_EVIDENCE: invalid boolean {column}")
    if not (
        candidates["exact_name_anchor"] | candidates["exact_address_anchor"]
    ).all():
        raise ValueError("STOP_V412_EVIDENCE: exact anchor missing")
    if not (
        candidates["strong_name_evidence"]
        & candidates["strong_address_evidence"]
    ).all():
        raise ValueError("STOP_V412_EVIDENCE: weak direct evidence")
    if candidates["direct_evidence_class"].ne("NAME_AND_ADDRESS").any():
        raise ValueError("STOP_V412_EVIDENCE: evidence class changed")
    expected_rules = [
        direct_match_rule(row)
        for row in candidates.to_dict("records")
    ]
    if candidates["direct_match_rule"].astype(str).tolist() != expected_rules:
        raise ValueError("STOP_V412_EVIDENCE: direct match rule changed")
    expected_refs = [
        evidence_ref(str(row.query_id), str(row.candidate_siret))
        for row in candidates.itertuples(index=False)
    ]
    if candidates["evidence_ref"].astype(str).tolist() != expected_refs:
        raise ValueError("STOP_V412_EVIDENCE: invalid evidence reference")

    grouped = {
        str(query_id): group
        for query_id, group in candidates.groupby("query_id", sort=False)
    }
    for query in queries.to_dict("records"):
        query_id = str(query["query_id"])
        rows = grouped.get(query_id, candidates.iloc[0:0])
        count = len(rows)
        sirens = rows["candidate_siren"].astype(str).tolist()
        expected = query_evidence_record(
            query_id=query_id,
            partition_key=str(query["partition_key"]),
            active_universe_count=int(query["active_universe_count"]),
            candidates=rows.to_dict("records"),
        )
        for column in QUERY_EVIDENCE_COLUMNS[3:]:
            observed = query[column]
            target = expected[column]
            if pd.isna(observed) and target is None:
                continue
            if observed != target:
                raise ValueError(
                    f"STOP_V412_EVIDENCE: aggregate mismatch {query_id}:{column}"
                )
        if count and len(set(sirens)) < 1:
            raise ValueError("STOP_V412_EVIDENCE: missing candidate SIREN")
    unknown = set(candidates["query_id"].astype(str)) - set(
        queries["query_id"].astype(str)
    )
    if unknown:
        raise ValueError("STOP_V412_EVIDENCE: candidate without query")


def apply_guard(
    *,
    decision_v411: str,
    review_reason_v411: str | None,
    predicted_siret: str | None,
    direct_candidate_count: int,
    sole_direct_siret: str | None,
) -> tuple[str, str | None]:
    """Apply V4.12-G as a veto; it can never create or replace a match."""

    if decision_v411 not in {"AUTO_MATCH", "REVIEW"}:
        raise ValueError("STOP_V412_GUARD: invalid V4.11 decision")
    if (
        not isinstance(direct_candidate_count, Integral)
        or isinstance(direct_candidate_count, (bool, np.bool_))
        or int(direct_candidate_count) < 0
    ):
        raise ValueError("STOP_V412_GUARD: invalid direct candidate count")
    direct_candidate_count = int(direct_candidate_count)
    if decision_v411 != "AUTO_MATCH":
        return "REVIEW", review_reason_v411
    if direct_candidate_count == 0:
        return "REVIEW", "NO_DIRECT_EVIDENCE"
    if direct_candidate_count >= 2:
        return "REVIEW", "MULTIPLE_STRONG_DIRECT_CANDIDATES"
    if (
        not isinstance(sole_direct_siret, str)
        or not _SIRET.fullmatch(sole_direct_siret)
        or not isinstance(predicted_siret, str)
        or not _SIRET.fullmatch(predicted_siret)
        or sole_direct_siret != predicted_siret
    ):
        return "REVIEW", "DIRECT_EVIDENCE_DISAGREES_TOP1"
    return "AUTO_MATCH", None


__all__ = [
    "CANDIDATE_EVIDENCE_COLUMNS",
    "GUARD_REASONS",
    "QUERY_EVIDENCE_COLUMNS",
    "apply_guard",
    "build_evidence_frames",
    "candidate_evidence_record",
    "direct_match_rule",
    "evidence_ref",
    "query_evidence_record",
    "validate_evidence",
]
