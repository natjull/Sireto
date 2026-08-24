"""Sealed exhaustive proof for the strongly projected v12 parent scanner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import duckdb
import pyarrow.parquet as pq

from .retrieval_top100_cascade_v12 import is_forbidden_input_key
from .retrieval_v12_parent_arrow import parent_signal_complete_projection_v12
from .retrieval_v12_scene_ir import SceneIRCache, sha256_file


PARENT_SIGNAL_CAPABILITY_SCHEMA_V12 = (
    "sireto-retrieval-v12-parent-signal-completeness-1"
)
_RESERVED_RANKS = frozenset(
    {"union_rank", "retrieval_rank", "parent_rank", "site_rank"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class ParentSignalCapabilityV12:
    manifest_path: Path
    build_id: str
    source_candidate_sha256: str
    source_manifest_sha256: str
    candidate_rows: int
    projection: tuple[str, ...]


def _identity(scene_cache: SceneIRCache) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    manifest = scene_cache.manifest
    source_sha = manifest.get("source_candidate_sha256")
    source_manifest_sha = manifest.get("source_manifest_sha256")
    if (
        not isinstance(source_sha, str)
        or _SHA256.fullmatch(source_sha) is None
        or not isinstance(source_manifest_sha, str)
        or _SHA256.fullmatch(source_manifest_sha) is None
    ):
        raise ValueError("SceneIR lacks sealed source hashes for parent capability")
    schema = pq.read_schema(scene_cache.source)
    safe_names = tuple(
        name for name in schema.names if not is_forbidden_input_key(name)
    )
    projection = parent_signal_complete_projection_v12(safe_names)
    rank_columns = tuple(
        name
        for name in safe_names
        if name.endswith("_rank") and name not in _RESERVED_RANKS
    )
    identity = {
        "schema_version": PARENT_SIGNAL_CAPABILITY_SCHEMA_V12,
        "scene_ir_build_id": manifest.get("build_id"),
        "source_candidate_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_candidate_rows": int(pq.ParquetFile(scene_cache.source).metadata.num_rows),
        "rank_columns": list(rank_columns),
        "parent_projection": list(projection),
        "proof": "EXHAUSTIVE_ALL_ROWS_CHANNELS_AND_NONNULL_RANKS_HAVE_SIGNAL",
    }
    return identity, projection, rank_columns


def build_parent_signal_capability_v12(
    scene_cache: SceneIRCache,
    *,
    output_root: Path,
    threads: int = 4,
) -> ParentSignalCapabilityV12:
    """Scan every source row and seal the signal-completeness capability."""

    if not 1 <= int(threads) <= 8:
        raise ValueError("parent capability threads must be in [1, 8]")
    identity, projection, rank_columns = _identity(scene_cache)
    build_id = hashlib.sha256(_canonical(identity)).hexdigest()
    root = Path(output_root).resolve()
    workspace = Path(__file__).resolve().parents[2]
    if root == workspace or workspace in root.parents:
        raise ValueError("parent capability output must be on external storage")
    target = root / build_id[:16]
    if target.exists():
        return load_parent_signal_capability_v12(target / "manifest.json", scene_cache)
    root.mkdir(parents=True, exist_ok=True)

    signal_channels = "list_transform(signals, item -> item.channel)"
    channel_violation = (
        "NOT (channels IS NULL OR len(channels) = 0 OR "
        f"coalesce(list_has_all({signal_channels}, channels), false))"
    )
    rank_violations = [
        "(" + _quote_identifier(name) + " IS NOT NULL AND NOT coalesce(" \
        + f"list_contains({signal_channels}, {_quote_literal(name[:-5])}), false))"
        for name in rank_columns
    ]
    any_violation = " OR ".join([channel_violation, *rank_violations])
    selected = ["channels", "signals", *rank_columns]
    select_sql = ", ".join(_quote_identifier(name) for name in selected)
    source_literal = _quote_literal(str(scene_cache.source))
    sql = f"""
        SELECT
            count(*)::BIGINT AS candidate_rows,
            count_if({channel_violation})::BIGINT AS channel_violations,
            count_if({any_violation})::BIGINT AS total_violations
        FROM (
            SELECT {select_sql}
            FROM read_parquet({source_literal})
        )
    """
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        row = connection.execute(sql).fetchone()
    finally:
        connection.close()
    assert row is not None
    candidate_rows, channel_violations, total_violations = map(int, row)
    if candidate_rows != int(identity["source_candidate_rows"]):
        raise ValueError("parent signal capability row-count drift")
    if channel_violations or total_violations:
        raise ValueError(
            "parent source is not signal-complete: "
            f"channels={channel_violations}, total={total_violations}"
        )

    stage = Path(tempfile.mkdtemp(prefix=".parent-signal-v12-", dir=root))
    manifest = {
        **identity,
        "build_id": build_id,
        "candidate_rows": candidate_rows,
        "channel_violations": channel_violations,
        "total_violations": total_violations,
        "signals_complete": True,
        "fold0_opened": False,
        "fold1_opened": False,
        "human_labels_opened": False,
    }
    path = stage / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage.rename(target)
    return load_parent_signal_capability_v12(target / "manifest.json", scene_cache)


def load_parent_signal_capability_v12(
    manifest_path: Path,
    scene_cache: SceneIRCache,
) -> ParentSignalCapabilityV12:
    """Validate and bind an exhaustive capability to one exact SceneIR source."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    identity, projection, rank_columns = _identity(scene_cache)
    build_id = hashlib.sha256(_canonical(identity)).hexdigest()
    if (
        raw.get("schema_version") != PARENT_SIGNAL_CAPABILITY_SCHEMA_V12
        or raw.get("build_id") != build_id
        or path.parent.name != build_id[:16]
        or any(raw.get(key) != value for key, value in identity.items())
        or raw.get("signals_complete") is not True
        or raw.get("candidate_rows") != identity["source_candidate_rows"]
        or raw.get("channel_violations") != 0
        or raw.get("total_violations") != 0
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("human_labels_opened") is not False
        or raw.get("rank_columns") != list(rank_columns)
        or raw.get("parent_projection") != list(projection)
    ):
        raise ValueError("parent signal capability manifest is invalid or unbound")
    return ParentSignalCapabilityV12(
        manifest_path=path,
        build_id=build_id,
        source_candidate_sha256=str(identity["source_candidate_sha256"]),
        source_manifest_sha256=str(identity["source_manifest_sha256"]),
        candidate_rows=int(identity["source_candidate_rows"]),
        projection=projection,
    )


__all__ = [
    "PARENT_SIGNAL_CAPABILITY_SCHEMA_V12",
    "ParentSignalCapabilityV12",
    "build_parent_signal_capability_v12",
    "load_parent_signal_capability_v12",
]
