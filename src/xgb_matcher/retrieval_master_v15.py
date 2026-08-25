"""Bounded, label-blind feasibility engine for the Master@2000 gate.

The module deliberately has no label loader.  It consumes an immutable v11
prefix, retrieves parent SIRENs from official indices, expands them through the
published site sidecar, and appends at most 250 new SIRETs.  Truth-side cohort
construction and evaluation live in a separate module/script.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .hierarchical_retrieval import RetrievalQuery


GENERATION_SCHEMA_VERSION = "sireto-master-v15-feasibility-generation-1"
QUERY_SCHEMA = pa.schema([pa.field("query_id", pa.string()), pa.field("fold", pa.int8())])
CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("fold", pa.int8()),
        pa.field("candidate_rank", pa.int16()),
        pa.field("siret", pa.string()),
        pa.field("source", pa.string()),
        pa.field("base_rank", pa.int16()),
        pa.field("parent_rank", pa.int16()),
        pa.field("site_rank", pa.int16()),
    ]
)
PARENT_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("fold", pa.int8()),
        pa.field("parent_rank", pa.int16()),
        pa.field("siren", pa.string()),
        pa.field("rrf_score", pa.float64()),
        pa.field("backend_support", pa.int8()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class MasterV15Recipe:
    base_prefix: int = 1750
    rescue_budget: int = 250
    final_cap: int = 2000
    backend_parent_limit: int = 64
    parent_cap: int = 512
    rrf_k: int = 60
    site_depth: int = 16

    def __post_init__(self) -> None:
        if self.base_prefix + self.rescue_budget != self.final_cap:
            raise ValueError("base prefix plus rescue budget must equal final cap")
        if self.final_cap != 2000:
            raise ValueError("Master v15 feasibility has an immutable cap of 2000")
        if not 1 <= self.backend_parent_limit <= 64:
            raise ValueError("backend parent limit exceeds the contract")
        if not 1 <= self.parent_cap <= 512:
            raise ValueError("global parent cap exceeds the contract")
        if not 1 <= self.site_depth <= 16:
            raise ValueError("site depth exceeds the contract")
        if self.rrf_k != 60:
            raise ValueError("RRF constant is frozen at 60")

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "MasterV15Recipe":
        return cls(**{key: int(value) for key, value in raw["recipe"].items()})


def load_query_selection(path: Path) -> tuple[tuple[str, int], ...]:
    table = pq.read_table(path)
    if set(table.column_names) != {"query_id", "fold"}:
        raise ValueError("generator query selection must contain only query_id and fold")
    rows = tuple((str(row["query_id"]), int(row["fold"])) for row in table.to_pylist())
    if not rows or len(rows) != len({query_id for query_id, _ in rows}):
        raise ValueError("query selection must be non-empty and query_id-unique")
    if any(not query_id or fold not in {2, 3, 4} for query_id, fold in rows):
        raise ValueError("query selection opened an invalid query or fold")
    return rows


def rrf_parents(
    backends: Sequence[tuple[str, Any]],
    query: RetrievalQuery,
    recipe: MasterV15Recipe,
) -> tuple[dict[str, Any], ...]:
    scores: dict[str, float] = {}
    support: dict[str, int] = {}
    sources: dict[str, list[str]] = {}
    for backend_name, backend in backends:
        seen: set[str] = set()
        for rank, (raw_siren, _score) in enumerate(
            backend.search_sirens(query, recipe.backend_parent_limit), start=1
        ):
            siren = str(raw_siren)
            if len(siren) != 9 or not siren.isdigit() or siren in seen:
                continue
            seen.add(siren)
            scores[siren] = scores.get(siren, 0.0) + 1.0 / (recipe.rrf_k + rank)
            support[siren] = support.get(siren, 0) + 1
            sources.setdefault(siren, []).append(str(backend_name))
    ordered = sorted(scores, key=lambda siren: (-scores[siren], -support[siren], siren))
    return tuple(
        {
            "siren": siren,
            "parent_rank": rank,
            "rrf_score": scores[siren],
            "backend_support": support[siren],
            "backend_names": tuple(sources[siren]),
        }
        for rank, siren in enumerate(ordered[: recipe.parent_cap], start=1)
    )


def compose_query(
    *,
    query_id: str,
    fold: int,
    source_rows: Iterable[Mapping[str, Any]],
    backends: Sequence[tuple[str, Any]],
    sidecars: Any,
    recipe: MasterV15Recipe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = sorted(source_rows, key=lambda row: (int(row["union_rank"]), str(row["siret"])))
    if not rows:
        raise ValueError("selected SceneIR query has no v11 candidates")
    first = rows[0]
    query = RetrievalQuery.from_mapping(first)
    if not query.insee and not query.postcode:
        parents: tuple[dict[str, Any], ...] = ()
    else:
        parents = rrf_parents(backends, query, recipe)

    base_rows = rows[: recipe.base_prefix]
    candidate_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in base_rows:
        siret = str(row["siret"])
        if not siret or siret in seen:
            continue
        seen.add(siret)
        candidate_rows.append(
            {
                "query_id": query_id,
                "fold": fold,
                "candidate_rank": len(candidate_rows) + 1,
                "siret": siret,
                "source": "BASE_V11_TOP1750",
                "base_rank": int(row["union_rank"]),
                "parent_rank": None,
                "site_rank": None,
            }
        )

    parent_rows = [
        {
            "query_id": query_id,
            "fold": fold,
            "parent_rank": int(parent["parent_rank"]),
            "siren": str(parent["siren"]),
            "rrf_score": float(parent["rrf_score"]),
            "backend_support": int(parent["backend_support"]),
        }
        for parent in parents
    ]
    if parents:
        expanded = sidecars.sites_by_sirens_geo(
            sirens=[str(parent["siren"]) for parent in parents],
            insee=query.insee,
            postcode=query.postcode,
            address=query.address,
            street_number=query.number,
            street="",
            include_active=True,
            include_closed=True,
            max_sites_per_siren=recipe.site_depth,
        )
        rescue: list[tuple[int, int, str]] = []
        for parent in parents:
            parent_rank = int(parent["parent_rank"])
            result = expanded.get(str(parent["siren"]))
            for site_rank, site in enumerate((result.sites if result else ()), start=1):
                rescue.append((site_rank, parent_rank, str(site.siret)))
        rescue.sort()
        emitted = 0
        for site_rank, parent_rank, siret in rescue:
            if siret in seen:
                continue
            seen.add(siret)
            emitted += 1
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "candidate_rank": len(candidate_rows) + 1,
                    "siret": siret,
                    "source": "OFFICIAL_PARENT_RRF_SITE_RESCUE",
                    "base_rank": None,
                    "parent_rank": parent_rank,
                    "site_rank": site_rank,
                }
            )
            if emitted >= recipe.rescue_budget:
                break
    if len(candidate_rows) > recipe.final_cap:
        raise AssertionError("Master v15 candidate cap exceeded")
    return candidate_rows, parent_rows


def write_generation(
    output_dir: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    parent_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    candidates_path = output_dir / "candidates.parquet"
    parents_path = output_dir / "parents.parquet"
    pq.write_table(pa.Table.from_pylist(list(candidate_rows), schema=CANDIDATE_SCHEMA), candidates_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(list(parent_rows), schema=PARENT_SCHEMA), parents_path, compression="zstd")
    payload = {
        **dict(manifest),
        "schema_version": GENERATION_SCHEMA_VERSION,
        "files": {
            "candidates.parquet": {"rows": len(candidate_rows), "size_bytes": candidates_path.stat().st_size, "sha256": sha256_file(candidates_path)},
            "parents.parquet": {"rows": len(parent_rows), "size_bytes": parents_path.stat().st_size, "sha256": sha256_file(parents_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


__all__ = [
    "CANDIDATE_SCHEMA", "GENERATION_SCHEMA_VERSION", "MasterV15Recipe",
    "PARENT_SCHEMA", "compose_query", "load_query_selection", "rrf_parents",
    "sha256_file", "write_generation",
]
