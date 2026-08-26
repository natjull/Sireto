"""Label-blind geographic features for the bounded v16 side quest.

BAN resolves the CRM address.  Official SIRENE Lambert-93 coordinates locate
candidate establishments.  Human truth is deliberately absent from this
module; it is only opened by ``retrieval_geo_v16_evaluation``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import JaroWinkler


SCHEMA_VERSION = "sireto-retrieval-geo-v16-generation-1"
ALLOWED_FOLDS = {2, 3, 4}
_GRS80_A = 6_378_137.0
_GRS80_F = 1.0 / 298.257222101
_GRS80_E = math.sqrt(2.0 * _GRS80_F - _GRS80_F * _GRS80_F)


def _lcc_t(latitude_rad: float) -> float:
    sin_lat = math.sin(latitude_rad)
    eccentric = ((1.0 - _GRS80_E * sin_lat) / (1.0 + _GRS80_E * sin_lat)) ** (_GRS80_E / 2.0)
    return math.tan(math.pi / 4.0 - latitude_rad / 2.0) / eccentric


def _lcc_m(latitude_rad: float) -> float:
    return math.cos(latitude_rad) / math.sqrt(1.0 - (_GRS80_E * math.sin(latitude_rad)) ** 2)


_L93_LAT1 = math.radians(49.0)
_L93_LAT2 = math.radians(44.0)
_L93_LAT0 = math.radians(46.5)
_L93_LON0 = math.radians(3.0)
_L93_N = (math.log(_lcc_m(_L93_LAT1)) - math.log(_lcc_m(_L93_LAT2))) / (
    math.log(_lcc_t(_L93_LAT1)) - math.log(_lcc_t(_L93_LAT2))
)
_L93_F = _lcc_m(_L93_LAT1) / (_L93_N * _lcc_t(_L93_LAT1) ** _L93_N)
_L93_RHO0 = _GRS80_A * _L93_F * _lcc_t(_L93_LAT0) ** _L93_N

QUERY_GEO_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("fold", pa.int8()),
        pa.field("ban_resolved", pa.bool_()),
        pa.field("ban_street_similarity", pa.float32()),
        pa.field("ban_geography_match", pa.string()),
        pa.field("ban_type_position", pa.string()),
        pa.field("ban_source_position", pa.string()),
        pa.field("ban_certification_commune", pa.bool_()),
        pa.field("ban_uncertainty_m", pa.float32()),
        pa.field("ban_declared_weak", pa.bool_()),
        pa.field("ban_match_count", pa.int32()),
    ]
)
QUERY_GEO_INTERNAL_SCHEMA = pa.schema(
    list(QUERY_GEO_SCHEMA)
    + [pa.field("ban_x_l93", pa.float64()), pa.field("ban_y_l93", pa.float64())]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", text).upper().split())


def normalize_street(address: Any, number: Any = "") -> str:
    normalized = normalize_text(address)
    number_norm = normalize_text(number)
    if number_norm and normalized.startswith(number_norm + " "):
        normalized = normalized[len(number_norm) + 1 :]
    return normalized


def street_similarity(left: Any, right: Any) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 0.0
    return float(JaroWinkler.normalized_similarity(a, b))


_POSITION_RADIUS_M = {
    "ENTREE": 8.0,
    "BATIMENT": 15.0,
    "LOGEMENT": 15.0,
    "PARCELLE": 30.0,
    "SEGMENT": 60.0,
    "CAGE D ESCALIER": 20.0,
    "DELIVRANCE POSTALE": 150.0,
    "SERVICE TECHNIQUE": 100.0,
    "AUTRE": 100.0,
    "INCONNUE": 250.0,
    "": 250.0,
}


def ban_uncertainty_m(
    *, type_position: Any, source_position: Any, certified: bool, street_score: float
) -> float:
    """Conservative uncertainty proxy, never an accuracy guarantee."""

    radius = _POSITION_RADIUS_M.get(normalize_text(type_position), 150.0)
    if normalize_text(source_position) in {"", "INCONNUE"}:
        radius *= 1.75
    if not certified:
        radius *= 1.35
    radius *= 1.0 + 2.0 * max(0.0, 0.9 - float(street_score))
    return float(min(1000.0, max(5.0, radius)))


def ban_declared_weak(
    *, type_position: Any, source_position: Any, certified: bool, street_score: float
) -> bool:
    return bool(
        float(street_score) < 0.84
        or not certified
        or normalize_text(source_position) in {"", "INCONNUE"}
        or normalize_text(type_position)
        in {"", "INCONNUE", "AUTRE", "DELIVRANCE POSTALE", "SERVICE TECHNIQUE"}
    )


def lonlat_to_l93(longitude: float, latitude: float) -> tuple[float, float]:
    lon, lat = math.radians(float(longitude)), math.radians(float(latitude))
    rho = _GRS80_A * _L93_F * _lcc_t(lat) ** _L93_N
    angle = _L93_N * (lon - _L93_LON0)
    x = 700_000.0 + rho * math.sin(angle)
    y = 6_600_000.0 + _L93_RHO0 - rho * math.cos(angle)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("BAN coordinate could not be transformed to Lambert-93")
    return float(x), float(y)


def l93_distance_m(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(float(x2) - float(x1), float(y2) - float(y1)))


@dataclass(frozen=True)
class BanCandidate:
    longitude: float
    latitude: float
    street: str
    postcode: str
    insee: str
    type_position: str
    source_position: str
    certified: bool
    geography_match: str


def choose_ban_match(
    *, crm_street: str, candidates: Sequence[BanCandidate]
) -> Mapping[str, Any] | None:
    """Choose deterministically without any truth-side information."""

    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        similarity = street_similarity(crm_street, candidate.street)
        geography_priority = 1 if candidate.geography_match == "INSEE" else 0
        scored.append(
            (
                -geography_priority,
                -similarity,
                -int(candidate.certified),
                normalize_text(candidate.type_position),
                normalize_text(candidate.source_position),
                candidate.longitude,
                candidate.latitude,
                candidate,
                similarity,
            )
        )
    *_, chosen, score = min(scored)
    x, y = lonlat_to_l93(chosen.longitude, chosen.latitude)
    uncertainty = ban_uncertainty_m(
        type_position=chosen.type_position,
        source_position=chosen.source_position,
        certified=chosen.certified,
        street_score=score,
    )
    return {
        "ban_resolved": True,
        "ban_x_l93": x,
        "ban_y_l93": y,
        "ban_street_similarity": score,
        "ban_geography_match": chosen.geography_match,
        "ban_type_position": chosen.type_position,
        "ban_source_position": chosen.source_position,
        "ban_certification_commune": chosen.certified,
        "ban_uncertainty_m": uncertainty,
        "ban_declared_weak": ban_declared_weak(
            type_position=chosen.type_position,
            source_position=chosen.source_position,
            certified=chosen.certified,
            street_score=score,
        ),
        "ban_match_count": len(candidates),
    }


def validate_label_blind_queries(path: Path) -> tuple[tuple[str, int], ...]:
    table = pq.read_table(path)
    if set(table.column_names) != {"query_id", "fold"}:
        raise ValueError("geo v16 generator accepts only query_id and fold")
    rows = tuple((str(row["query_id"]), int(row["fold"])) for row in table.to_pylist())
    if not rows or len(rows) != len({row[0] for row in rows}):
        raise ValueError("query selection must be non-empty and query_id-unique")
    if any(not query_id or fold not in ALLOWED_FOLDS for query_id, fold in rows):
        raise ValueError("geo v16 may open folds 2/3/4 only")
    return rows


def write_manifest(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    root = Path(output_dir)
    files = {}
    for name in (
        "query_geo.parquet",
        "candidate_geo_features.parquet",
        "geo_neighbor_candidates.parquet",
        "geo_neighbor_parents.parquet",
    ):
        path = root / name
        files[name] = {
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        **dict(payload),
        "schema_version": SCHEMA_VERSION,
        "truth_columns_opened": False,
        "positive_injection": False,
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "BanCandidate",
    "QUERY_GEO_SCHEMA",
    "QUERY_GEO_INTERNAL_SCHEMA",
    "SCHEMA_VERSION",
    "ban_declared_weak",
    "ban_uncertainty_m",
    "choose_ban_match",
    "l93_distance_m",
    "lonlat_to_l93",
    "normalize_street",
    "normalize_text",
    "sha256_file",
    "street_similarity",
    "validate_label_blind_queries",
    "write_manifest",
]
