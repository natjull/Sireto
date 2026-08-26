"""Bounded v17 parent discovery and exhaustive SIREN-to-SIRET primitives.

This module is deliberately truth-free.  Human labels are consumed only by
the separate evaluation script after the generated artifacts are sealed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq


GENERATION_SCHEMA_VERSION = "sireto-retrieval-v17-site-inventory-generation-1"
EVALUATION_SCHEMA_VERSION = "sireto-retrieval-v17-site-inventory-evaluation-1"
ALLOWED_FOLDS = {2, 3, 4}
SOURCE_ORDER = ("TEXT", "ADDRESS", "GEO")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_identifier(value: Any, width: int) -> str:
    text = "".join(character for character in str(value or "") if character.isdigit())
    return text if len(text) == width else ""


_SUFFIXES = {
    "B": "BIS",
    "BIS": "BIS",
    "T": "TER",
    "TER": "TER",
    "Q": "QUATER",
    "QUATER": "QUATER",
}


def split_street_number(value: Any) -> tuple[str, str]:
    """Return canonical number/suffix; zero and non-numeric values are absent."""

    match = re.match(r"^\s*0*(\d+)\s*([A-Za-z]+)?", str(value or ""))
    if match is None:
        return "", ""
    number = str(int(match.group(1)))
    if number == "0":
        return "", ""
    suffix = _SUFFIXES.get(str(match.group(2) or "").upper(), str(match.group(2) or "").upper())
    return number, suffix


def same_site_evidence(
    *,
    query_number: Any,
    query_street: Any,
    query_insee: Any,
    query_postcode: Any,
    site_number: Any,
    site_suffix: Any,
    site_street: Any,
    site_insee: Any,
    site_postcode: Any,
) -> bool:
    """Strict operational same-site evidence independent from any model score."""

    number, suffix = split_street_number(query_number)
    candidate_number, candidate_suffix = split_street_number(
        f"{site_number or ''} {site_suffix or ''}"
    )
    street = " ".join(str(query_street or "").upper().split())
    candidate_street = " ".join(str(site_street or "").upper().split())
    insee = str(query_insee or "")
    postcode = str(query_postcode or "")
    geography = bool(
        (insee and insee == str(site_insee or ""))
        or (postcode and postcode == str(site_postcode or ""))
    )
    return bool(
        number
        and candidate_number == number
        and (not suffix or candidate_suffix == suffix)
        and street
        and candidate_street == street
        and geography
    )


def validate_query_selection(path: Path, *, expected_count: int | None = None) -> tuple[tuple[str, int], ...]:
    table = pq.read_table(path)
    if set(table.column_names) != {"query_id", "fold"}:
        raise ValueError("v17 generator accepts only query_id and fold")
    rows = tuple((str(row["query_id"]), int(row["fold"])) for row in table.to_pylist())
    if not rows or len(rows) != len({query_id for query_id, _fold in rows}):
        raise ValueError("v17 query selection must be non-empty and unique")
    if any(not query_id or fold not in ALLOWED_FOLDS for query_id, fold in rows):
        raise ValueError("v17 opened a forbidden query or fold")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError("v17 query population drifted")
    return rows


def _rank_map(values: Sequence[str]) -> dict[str, int]:
    return {value: rank for rank, value in enumerate(dict.fromkeys(values), start=1)}


def rrf_parent_order(
    source_orders: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float],
    cap: int,
    rrf_k: int = 60,
) -> tuple[str, ...]:
    if not 1 <= cap <= 512 or rrf_k != 60:
        raise ValueError("v17 RRF requires cap<=512 and k=60")
    unknown = set(source_orders) - set(SOURCE_ORDER)
    if unknown:
        raise ValueError(f"unknown parent sources: {sorted(unknown)}")
    scores: dict[str, float] = {}
    support: dict[str, int] = {}
    best_rank: dict[str, int] = {}
    for source in SOURCE_ORDER:
        rank_map = _rank_map(tuple(source_orders.get(source, ())))
        weight = float(weights.get(source, 1.0))
        if weight <= 0:
            raise ValueError("source weights must be positive")
        for siren, rank in rank_map.items():
            if normalize_identifier(siren, 9) != siren:
                continue
            scores[siren] = scores.get(siren, 0.0) + weight / (rrf_k + rank)
            support[siren] = support.get(siren, 0) + 1
            best_rank[siren] = min(best_rank.get(siren, rank), rank)
    ordered = sorted(
        scores,
        key=lambda siren: (-scores[siren], -support[siren], best_rank[siren], siren),
    )
    return tuple(ordered[:cap])


def balanced_parent_order(
    source_orders: Mapping[str, Sequence[str]],
    *,
    quotas: Mapping[str, int],
    cap: int = 512,
    rrf_k: int = 60,
) -> tuple[str, ...]:
    if set(quotas) != set(SOURCE_ORDER) or sum(int(quotas[key]) for key in SOURCE_ORDER) != cap:
        raise ValueError("balanced quotas must exactly partition the parent cap")
    selected: list[str] = []
    seen: set[str] = set()
    for source in SOURCE_ORDER:
        emitted = 0
        for siren in source_orders.get(source, ()):
            if normalize_identifier(siren, 9) != siren or siren in seen:
                continue
            seen.add(siren)
            selected.append(siren)
            emitted += 1
            if emitted == int(quotas[source]):
                break
    if len(selected) < cap:
        fill = rrf_parent_order(
            source_orders,
            weights={source: 1.0 for source in SOURCE_ORDER},
            cap=cap,
            rrf_k=rrf_k,
        )
        for siren in fill:
            if siren not in seen:
                seen.add(siren)
                selected.append(siren)
                if len(selected) == cap:
                    break
    return tuple(selected)


def source_bitmap(siren: str, source_orders: Mapping[str, Sequence[str]]) -> tuple[bool, bool, bool]:
    members = {source: siren in set(source_orders.get(source, ())) for source in SOURCE_ORDER}
    return members["TEXT"], members["ADDRESS"], members["GEO"]


def build_parent_rows(
    *,
    query_id: str,
    fold: int,
    source_orders: Mapping[str, Sequence[str]],
    variants: Mapping[str, Mapping[str, Any]],
    primary_variant: str,
    cap: int = 512,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build preregistered candidates and unranked union presence."""

    all_sirens = set().union(*(set(source_orders.get(source, ())) for source in SOURCE_ORDER))
    presence = []
    source_sets = {source: set(source_orders.get(source, ())) for source in SOURCE_ORDER}
    for siren in sorted(all_sirens):
        if normalize_identifier(siren, 9) != siren:
            continue
        presence.append(
            {
                "query_id": query_id,
                "fold": fold,
                "siren": siren,
                "from_text": siren in source_sets["TEXT"],
                "from_address": siren in source_sets["ADDRESS"],
                "from_geo": siren in source_sets["GEO"],
            }
        )
    candidates: list[dict[str, Any]] = []
    for name, recipe in variants.items():
        if {"text_quota", "address_quota", "geo_quota"} <= set(recipe):
            order = balanced_parent_order(
                source_orders,
                quotas={
                    "TEXT": int(recipe["text_quota"]),
                    "ADDRESS": int(recipe["address_quota"]),
                    "GEO": int(recipe["geo_quota"]),
                },
                cap=cap,
            )
        else:
            order = rrf_parent_order(
                source_orders,
                weights={
                    "TEXT": float(recipe["text_weight"]),
                    "ADDRESS": float(recipe["address_weight"]),
                    "GEO": float(recipe["geo_weight"]),
                },
                cap=cap,
            )
        for rank, siren in enumerate(order, start=1):
            candidates.append(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "variant": name,
                    "is_primary": name == primary_variant,
                    "parent_rank": rank,
                    "siren": siren,
                    "from_text": siren in source_sets["TEXT"],
                    "from_address": siren in source_sets["ADDRESS"],
                    "from_geo": siren in source_sets["GEO"],
                }
            )
    return candidates, presence


def write_generation_manifest(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    root = Path(output_dir)
    files: dict[str, Any] = {}
    for name in (
        "address_parent_signals.parquet",
        "parent_presence.parquet",
        "parent_candidates.parquet",
        "expanded_all_sites.parquet",
    ):
        path = root / name
        parquet = pq.ParquetFile(path)
        files[name] = {
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        **dict(payload),
        "schema_version": GENERATION_SCHEMA_VERSION,
        "truth_columns_opened": False,
        "positive_injection": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "pre_score_site_cap": None,
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def require_complete_site_expansion(rows: Iterable[Mapping[str, Any]]) -> None:
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        key = str(row["query_id"]), str(row["siren"])
        emitted = int(row["emitted_site_count"])
        source = int(row["source_site_count"])
        if key in counts and counts[key] != (emitted, source):
            raise ValueError("site-count proof drifted within a parent group")
        counts[key] = emitted, source
    if not counts or any(emitted != source or source <= 0 for emitted, source in counts.values()):
        raise ValueError("v17 expansion is not exhaustive")


__all__ = [
    "ALLOWED_FOLDS",
    "EVALUATION_SCHEMA_VERSION",
    "GENERATION_SCHEMA_VERSION",
    "SOURCE_ORDER",
    "balanced_parent_order",
    "build_parent_rows",
    "normalize_identifier",
    "require_complete_site_expansion",
    "rrf_parent_order",
    "same_site_evidence",
    "sha256_file",
    "split_street_number",
    "validate_query_selection",
    "write_generation_manifest",
]
