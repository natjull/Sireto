#!/usr/bin/env python3
"""Build the immutable, label-side-only all-human v12 query ledger.

Only folds 2/3/4 are materialised.  Fold 0 is deliberately not part of the
v12 development population and the fold-1 artifact is never opened.  The
ledger is independent from candidate groups, so a labelled query remains in
the denominator even when retrieval produces zero candidates for it.

The preferred input is an upstream all-human ``benchmark.parquet``.  A sealed
v11 benchmark directory is also accepted: only its development exact,
operational and missing-label files are predicate-scanned, and only folds
2/3/4 are materialised.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SCHEMA_VERSION = "sireto-retrieval-all-human-v12-label-ledger-1"
CONFIG_SCHEMA_VERSION = "sireto-retrieval-all-human-v12-config-1"
V11_SCHEMA_VERSION = "sireto-retrieval-all-human-v11-benchmark-1"
RAW_ALL_HUMAN_SCHEMA_VERSION = "sireto-crm-gt-v2-retrieval-input-3"
POPULATION = "ALL_USABLE_HUMAN_LABELS"
ONLY_EXCLUSION = "NO_VALID_CURRENT_OR_HISTORICAL_HUMAN_SIRET"
OPERATIONAL_EQUIVALENCE_MODE = (
    "EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE"
)
RUNTIME_QUERY_SAMPLE_RULE = "SHA256_QUERY_ID_ASC_FIRST_512"
RUNTIME_QUERY_SAMPLE_SIZE = 512
OOF_FOLDS = (2, 3, 4)
SIRET_RE = re.compile(r"^[0-9]{14}$")

LABEL_LEDGER_COLUMNS = (
    "query_id",
    "fold",
    "human_ground_truth_siret",
    "human_ground_truth_siren",
    "human_ground_truth_source",
    "ground_truth_state",
    "label_kind",
    "qualification_v2",
    "qualification_v3",
    "source_kind",
    "data_origin",
    "split",
    "unseen_siren",
    "new_site_known_siren",
    "mega_base_pool",
    "pool_size",
    "insee",
    "official_catalog_present",
)
OPERATIONAL_COLUMNS = (
    "query_id",
    "acceptable_sirets_operational_json",
)
EXCLUDED_COLUMNS = (
    "query_id",
    "fold",
    "missing_reason",
    "current_siret_present_but_invalid",
    "historical_siret_present_but_invalid",
    "label_kind",
    "qualification_v2",
    "qualification_v3",
    "source_kind",
    "data_origin",
    "split",
)

_RAW_OPTIONAL_COLUMNS = (
    "acceptable_sirets_operational",
    "acceptable_sirets_operational_json",
    "ground_truth_state",
    "label_kind",
    "qualification_v2",
    "qualification_v3",
    "source_kind",
    "data_origin",
    "split",
    "unseen_siren",
    "new_site_known_siren",
    "mega_base_pool",
    "pool_size",
    "insee",
    "is_synthetic",
)
_FORBIDDEN_EXACT_COLUMNS = {
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "union_rank",
    "hydration_rank",
    "selected_rank",
    "parent_rank",
    "site_rank",
    "ltr_score",
    "model_score",
    "final_score",
    "exact_hit_at_100",
    "exact_oracle_hit",
    "oracle_hit",
}


@dataclass(frozen=True)
class SourceLayout:
    kind: str
    manifest_path: Path
    exact_path: Path
    exact_sha256: str
    operational_path: Path | None = None
    operational_sha256: str | None = None
    excluded_path: Path | None = None
    excluded_sha256: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_siret(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if SIRET_RE.fullmatch(text) else ""


def parse_operational(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set)):
        try:
            if bool(pd.isna(value)):
                return ()
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("operational SIRETs are not valid JSON") from exc
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("operational SIRETs must be a JSON/list collection")
    return tuple(sorted({normalize_siret(item) for item in value} - {""}))


def _expected_hash(manifest: Mapping[str, Any], path: Path) -> str | None:
    for container_name in ("outputs", "files", "output_sha256"):
        container = manifest.get(container_name)
        if not isinstance(container, Mapping):
            continue
        direct = container.get(path.name)
        if isinstance(direct, str):
            return direct
        if isinstance(direct, Mapping):
            value = direct.get("sha256")
            if isinstance(value, str):
                return value
        for metadata in container.values():
            if not isinstance(metadata, Mapping):
                continue
            declared_path = metadata.get("path")
            if declared_path == path.name or Path(str(declared_path)).name == path.name:
                value = metadata.get("sha256")
                if isinstance(value, str):
                    return value
    return None


def _output_metadata(
    manifest: Mapping[str, Any], path: Path
) -> Mapping[str, Any] | None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return None
    direct = outputs.get(path.name)
    if isinstance(direct, Mapping):
        return direct
    for metadata in outputs.values():
        if isinstance(metadata, Mapping) and Path(
            str(metadata.get("path", ""))
        ).name == path.name:
            return metadata
    return None


def _pin_input(manifest: Mapping[str, Any], path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Missing source artifact: {path.name}")
    expected = _expected_hash(manifest, path)
    if not expected:
        raise ValueError(f"Source manifest does not pin {path.name}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"Source hash mismatch: {path.name}")
    return observed


def _validate_prepared_raw_lineage(
    manifest: Mapping[str, Any], source: Path
) -> Path:
    """Accept only the historical, sealed all-human input lineage.

    This is deliberately stricter than trusting ``population_scope`` alone:
    the schema and both independent no-model/no-injection declarations are
    part of the historical producer contract.  Its development file excludes
    fold 1 physically, so the all-fold benchmark is not selected.
    """

    identity = manifest.get("build_identity") or {}
    if manifest.get("schema_version") != RAW_ALL_HUMAN_SCHEMA_VERSION:
        raise ValueError("Raw source lacks an accepted all-human lineage schema")
    if identity.get("population_scope") != "all_human":
        raise ValueError("Raw source is not the complete all-human lineage")
    if identity.get("positive_injection") is not False:
        raise ValueError("Raw all-human lineage does not forbid positive injection")
    if identity.get("qualification_uses_retrieval_or_model_scores") is not False:
        raise ValueError("Raw all-human lineage may have used retrieval/model outputs")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Raw source does not independently forbid positive injection")
    if manifest.get("qualification_uses_retrieval_or_model_scores") is not False:
        raise ValueError("Raw source does not independently forbid model qualification")
    if manifest.get("test_locked") is not True:
        raise ValueError("Raw all-human source does not seal fold 1")
    if tuple(manifest.get("development_opened_folds") or ()) != (0, 2, 3, 4):
        raise ValueError("Raw all-human development lineage has unexpected folds")
    for key in (
        "builder_sha256",
        "population_manifest_sha256",
        "retrieval_contract_sha256",
        "partitions_sha256",
    ):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Raw all-human lineage lacks pinned {key}")
    query_count = manifest.get("query_count")
    split_counts = manifest.get("split_counts")
    if not isinstance(query_count, int) or query_count < 0:
        raise ValueError("Raw all-human lineage lacks its query count")
    if not isinstance(split_counts, Mapping) or not all(
        isinstance(value, int) and value >= 0 for value in split_counts.values()
    ):
        raise ValueError("Raw all-human lineage lacks split counts")
    if sum(split_counts.values()) != query_count or "crm_train" not in split_counts:
        raise ValueError("Raw all-human lineage split counts do not reconcile")
    development = source / "development.parquet"
    if not development.is_file():
        raise ValueError("Raw all-human lineage lacks its fold-1-free development file")
    if _expected_hash(manifest, development) is None:
        raise ValueError("Raw all-human lineage does not pin development.parquet")
    # The locked test file is deliberately neither opened nor hashed here.
    locked_hash = _expected_hash(manifest, source / "test_locked.parquet")
    if not isinstance(locked_hash, str) or len(locked_hash) != 64:
        raise ValueError("Raw all-human lineage does not pin its sealed fold-1 file")
    expected_development_rows = query_count - int(
        split_counts.get("crm_prospective_test", 0)
    )
    if pq.ParquetFile(development).metadata.num_rows != expected_development_rows:
        raise ValueError("Raw all-human development file is an arbitrary subset")
    return development


def _validate_explicit_raw_claims(
    manifest: Mapping[str, Any], source: Path
) -> Path:
    """Accept a future raw ledger only with explicit all-human/fold claims."""

    required = {
        "population": POPULATION,
        "only_exclusion": ONLY_EXCLUSION,
        "positive_injection": False,
        "retrieval_outputs_read": False,
        "model_outputs_read": False,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Raw source lacks explicit ALL_USABLE population lineage")
    if tuple(manifest.get("folds") or ()) != OOF_FOLDS:
        raise ValueError("Explicit raw source must already contain only folds 2/3/4")
    query_count = manifest.get("query_count")
    excluded_count = manifest.get("excluded_count")
    counts_by_fold = manifest.get("counts_by_fold")
    if not isinstance(query_count, int) or query_count < 0 or excluded_count != 0:
        raise ValueError("Explicit raw source lacks complete population counts")
    if not isinstance(counts_by_fold, Mapping) or set(counts_by_fold) != {
        "2",
        "3",
        "4",
    }:
        raise ValueError("Explicit raw source lacks exact per-fold counts")
    if any(
        not isinstance(value, int) or value < 0 for value in counts_by_fold.values()
    ) or sum(counts_by_fold.values()) != query_count:
        raise ValueError("Explicit raw source per-fold counts do not reconcile")
    benchmark = source / "benchmark.parquet"
    if not benchmark.is_file() or _expected_hash(manifest, benchmark) is None:
        raise ValueError("Explicit raw source does not pin benchmark.parquet")
    return benchmark


def _validate_v11_lineage(manifest: Mapping[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != V11_SCHEMA_VERSION:
        raise ValueError("Sealed v11 source schema is not the preregistered builder schema")
    identity = manifest.get("build_identity") or {}
    integrity = manifest.get("integrity") or {}
    fold_lock = manifest.get("fold_lock") or {}
    metrics = manifest.get("metrics") or {}
    population = manifest.get("population") or {}
    expected_identity = {
        "only_exclusion": ONLY_EXCLUSION,
        "positive_injection": False,
        "development_folds": [0, 2, 3, 4],
        "test_locked_fold": 1,
        "label_precedence": [
            "ground_truth_siret",
            "historical_ground_truth_siret",
        ],
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Sealed v11 source build identity is incomplete")
    expected_integrity = {
        "positive_injection": False,
        "retrieval_or_model_outputs_read": False,
        "synthetic_rows": 0,
        "operational_labels_physically_separate": True,
        "only_missing_labels_excluded": True,
    }
    if any(integrity.get(key) != value for key, value in expected_integrity.items()):
        raise ValueError("Sealed v11 source integrity claims are incomplete")
    expected_fold_lock = {
        "development_folds": [0, 2, 3, 4],
        "test_fold": 1,
        "fold1_locked": True,
        "fold1_opened": False,
        "development_contains_fold1": False,
        "test_consumption_requires_dedicated_authorization": True,
    }
    if any(fold_lock.get(key) != value for key, value in expected_fold_lock.items()):
        raise ValueError("Sealed v11 source fold lock is incomplete")
    if metrics.get("primary") != "EXACT_SIRET_ALL_USABLE_HUMAN_LABELS":
        raise ValueError("Sealed v11 source is not the all-usable-human population")
    source_rows = population.get("source_rows")
    usable = population.get("usable_human_labels")
    excluded = population.get("excluded_missing_human_labels")
    if not all(isinstance(value, int) and value >= 0 for value in (source_rows, usable, excluded)):
        raise ValueError("Sealed v11 source population counts are missing")
    if usable + excluded != source_rows:
        raise ValueError("Sealed v11 source population accounting is inconsistent")
    usable_by_fold = population.get("usable_by_fold") or {}
    missing_by_fold = population.get("missing_by_fold") or {}
    expected_fold_keys = {"0", "1", "2", "3", "4"}
    if set(usable_by_fold) != expected_fold_keys or set(missing_by_fold) != expected_fold_keys:
        raise ValueError("Sealed v11 source lacks complete per-fold accounting")
    if sum(int(value) for value in usable_by_fold.values()) != usable:
        raise ValueError("Sealed v11 usable counts do not reconcile")
    if sum(int(value) for value in missing_by_fold.values()) != excluded:
        raise ValueError("Sealed v11 excluded counts do not reconcile")

    expected_outputs = {
        "development.parquet": ([0, 2, 3, 4], False, sum(int(usable_by_fold[str(f)]) for f in (0, 2, 3, 4))),
        "operational_labels_development.parquet": ([0, 2, 3, 4], False, sum(int(usable_by_fold[str(f)]) for f in (0, 2, 3, 4))),
        "missing_human_labels_development.parquet": ([0, 2, 3, 4], False, sum(int(missing_by_fold[str(f)]) for f in (0, 2, 3, 4))),
        "test_locked.parquet": ([1], True, int(usable_by_fold["1"])),
        "operational_labels_test_locked.parquet": ([1], True, int(usable_by_fold["1"])),
        "missing_human_labels_test_locked.parquet": ([1], True, int(missing_by_fold["1"])),
    }
    for filename, (folds, locked, rows) in expected_outputs.items():
        metadata = _output_metadata(manifest, source / filename)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Sealed v11 source lacks output declaration: {filename}")
        if metadata.get("folds") != folds or metadata.get("locked") is not locked:
            raise ValueError(f"Sealed v11 output lock/fold drift: {filename}")
        if metadata.get("rows") != rows:
            raise ValueError(f"Sealed v11 output population count drift: {filename}")
        sha = metadata.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"Sealed v11 output is not hash-pinned: {filename}")
        path = source / filename
        if not locked and (
            not path.is_file() or pq.ParquetFile(path).metadata.num_rows != rows
        ):
            raise ValueError(f"Sealed v11 development output is a subset: {filename}")


def _resolve_source(source: Path) -> SourceLayout:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Source must contain manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("positive_injection") is True or (
        isinstance(manifest.get("integrity"), Mapping)
        and manifest["integrity"].get("positive_injection") is True
    ):
        raise ValueError("Source declares positive injection")

    raw_path = source / "benchmark.parquet"
    if raw_path.is_file():
        if manifest.get("schema_version") == RAW_ALL_HUMAN_SCHEMA_VERSION:
            selected_raw_path = _validate_prepared_raw_lineage(manifest, source)
            source_kind = "SEALED_RAW_ALL_HUMAN_DEVELOPMENT"
        else:
            selected_raw_path = _validate_explicit_raw_claims(manifest, source)
            source_kind = "EXPLICIT_RAW_ALL_HUMAN_OOF"
        return SourceLayout(
            kind=source_kind,
            manifest_path=manifest_path,
            exact_path=selected_raw_path,
            exact_sha256=_pin_input(manifest, selected_raw_path),
        )

    exact_path = source / "development.parquet"
    operational_path = source / "operational_labels_development.parquet"
    excluded_path = source / "missing_human_labels_development.parquet"
    if exact_path.is_file():
        _validate_v11_lineage(manifest, source)
        return SourceLayout(
            kind="SEALED_V11_DEVELOPMENT",
            manifest_path=manifest_path,
            exact_path=exact_path,
            exact_sha256=_pin_input(manifest, exact_path),
            operational_path=operational_path,
            operational_sha256=_pin_input(manifest, operational_path),
            excluded_path=excluded_path,
            excluded_sha256=_pin_input(manifest, excluded_path),
        )
    raise ValueError(
        "Source must contain benchmark.parquet or the sealed v11 development artifacts"
    )


def _schema_names(path: Path) -> tuple[str, ...]:
    return tuple(pq.ParquetFile(path).schema_arrow.names)


def _assert_no_model_or_retrieval_columns(path: Path) -> None:
    names = {name.lower() for name in _schema_names(path)}
    forbidden = names.intersection(_FORBIDDEN_EXACT_COLUMNS)
    forbidden.update(
        name
        for name in names
        if name.startswith("candidate_")
        or name.startswith("retrieval_")
        or name.startswith("prediction_")
        or name.endswith("_model_score")
    )
    if forbidden:
        raise ValueError(
            "Retrieval/model output entered the label-side source: "
            f"{sorted(forbidden)}"
        )


def _fold_column(path: Path) -> str:
    names = set(_schema_names(path))
    if "oof_fold" in names:
        return "oof_fold"
    if "fold" in names:
        return "fold"
    raise ValueError(f"{path.name} has no fold column")


def _scan_oof_only(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Predicate-scan only folds 2/3/4; fold 0/1 rows are never materialised."""

    fold_column = _fold_column(path)
    available = set(_schema_names(path))
    selected = None
    if columns is not None:
        selected = [name for name in columns if name in available]
        if fold_column not in selected:
            selected.append(fold_column)
    table = ds.dataset(path, format="parquet").to_table(
        columns=selected,
        filter=ds.field(fold_column).isin(list(OOF_FOLDS)),
    )
    frame = table.to_pandas()
    if fold_column != "fold":
        frame = frame.rename(columns={fold_column: "fold"})
    if not frame.empty:
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        if not set(frame["fold"]).issubset(set(OOF_FOLDS)):
            raise AssertionError("fold 0/1 materialised despite the v12 predicate")
    return frame


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported v12 config schema")
    folds = config.get("folds") or {}
    labels = config.get("labels") or {}
    development = config.get("development") or {}
    if tuple(folds.get("outer_oof") or ()) != OOF_FOLDS:
        raise ValueError("v12 outer OOF folds must be exactly 2/3/4")
    if folds.get("development_contaminated") != 0 or folds.get("test_locked") != 1:
        raise ValueError("v12 fold locks drifted")
    if tuple(labels.get("precedence") or ()) != (
        "ground_truth_siret",
        "historical_ground_truth_siret",
    ):
        raise ValueError("v12 human label precedence drifted")
    if labels.get("only_exclusion") != ONLY_EXCLUSION:
        raise ValueError("v12 only-exclusion policy drifted")
    if labels.get("qualification_fields_may_filter_primary_population") is not False:
        raise ValueError("v12 qualification fields may not filter the population")
    if labels.get("operational_equivalence_mode") != OPERATIONAL_EQUIVALENCE_MODE:
        raise ValueError("v12 operational equivalence mode drifted")
    if development.get("fold0_reads") != 0:
        raise ValueError("v12 fold0 read policy drifted")
    if development.get("fold1_reads_before_explicit_authorization") != 0:
        raise ValueError("v12 fold1 read policy drifted")


def _base_label_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "query_id",
        "fold",
        "ground_truth_siret",
        "historical_ground_truth_siret",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Source benchmark missing columns: {missing}")
    if raw["query_id"].isna().any() or raw["query_id"].astype(str).duplicated().any():
        raise ValueError("query_id must be non-null and unique in folds 2/3/4")
    raw = raw.copy()
    raw["query_id"] = raw["query_id"].astype(str)
    if "is_synthetic" in raw and raw["is_synthetic"].fillna(False).astype(bool).any():
        raise ValueError("Synthetic row entered the all-human v12 ledger")

    current = raw["ground_truth_siret"].map(normalize_siret)
    historical = raw["historical_ground_truth_siret"].map(normalize_siret)
    raw["human_ground_truth_siret"] = current.where(current.ne(""), historical)
    raw["human_ground_truth_source"] = "CURRENT"
    raw.loc[current.eq("") & historical.ne(""), "human_ground_truth_source"] = (
        "HISTORICAL_FALLBACK"
    )
    raw["human_ground_truth_siren"] = raw["human_ground_truth_siret"].str[:9]

    usable = raw[raw["human_ground_truth_siret"].ne("")].copy()
    excluded = raw[raw["human_ground_truth_siret"].eq("")].copy()
    excluded["missing_reason"] = ONLY_EXCLUSION
    excluded["current_siret_present_but_invalid"] = (
        excluded["ground_truth_siret"].fillna("").astype(str).str.strip().ne("")
    )
    excluded["historical_siret_present_but_invalid"] = (
        excluded["historical_ground_truth_siret"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    if len(usable) + len(excluded) != len(raw):
        raise AssertionError("v12 population accounting failed")
    return usable, excluded


def _empty_default(column: str, length: int) -> pd.Series:
    if column in {
        "current_siret_present_but_invalid",
        "historical_siret_present_but_invalid",
        "unseen_siren",
        "new_site_known_siren",
        "official_catalog_present",
    }:
        return pd.Series([pd.NA] * length, dtype="boolean")
    if column in {"mega_base_pool", "pool_size"}:
        return pd.Series([pd.NA] * length, dtype="Int64")
    return pd.Series([""] * length, dtype="string")


def _project(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = _empty_default(column, len(output)).to_numpy()
    output = output[list(columns)].copy()
    output["query_id"] = output["query_id"].astype(str)
    if "fold" in output:
        output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    for column in (
        "unseen_siren",
        "new_site_known_siren",
        "official_catalog_present",
    ):
        if column in output:
            output[column] = output[column].astype("boolean")
    for column in ("mega_base_pool", "pool_size"):
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(
                "Int64"
            )
    for column in output.columns:
        if column not in {
            "fold",
            "current_siret_present_but_invalid",
            "historical_siret_present_but_invalid",
            "unseen_siren",
            "new_site_known_siren",
            "official_catalog_present",
            "mega_base_pool",
            "pool_size",
        }:
            output[column] = output[column].fillna("").astype(str)
    return output.sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _exact_only_operational(
    ledger: pd.DataFrame,
    upstream_values: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a conservative sidecar and audit unverified upstream siblings.

    Same-SIREN alone does not prove that two establishments are the same
    physical site.  Until a separately sealed same-site evidence sidecar is
    available, v12 retains only the exact human SIRET as operationally
    acceptable and never neutralises an upstream sibling during fitting.
    """

    upstream_values = upstream_values or {}
    counts_by_fold = {str(fold): 0 for fold in OOF_FOLDS}
    rows: list[dict[str, str]] = []
    for row in ledger.to_dict("records"):
        query_id = str(row["query_id"])
        exact = str(row["human_ground_truth_siret"])
        upstream = set(parse_operational(upstream_values.get(query_id)))
        unverified = upstream - {exact}
        counts_by_fold[str(int(row["fold"]))] += len(unverified)
        rows.append(
            {
                "query_id": query_id,
                "acceptable_sirets_operational_json": json.dumps(
                    [exact], separators=(",", ":")
                ),
            }
        )
    return _project(pd.DataFrame(rows), OPERATIONAL_COLUMNS), {
        "operational_equivalence_mode": OPERATIONAL_EQUIVALENCE_MODE,
        "unverified_operational_alternative_count": int(
            sum(counts_by_fold.values())
        ),
        "unverified_operational_alternative_count_by_fold": counts_by_fold,
    }


def _operational_from_raw(
    usable: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_column = next(
        (
            name
            for name in (
                "acceptable_sirets_operational_json",
                "acceptable_sirets_operational",
            )
            if name in usable.columns
        ),
        None,
    )
    upstream = (
        {
            str(row["query_id"]): row.get(source_column)
            for row in usable.to_dict("records")
        }
        if source_column
        else {}
    )
    return _exact_only_operational(usable, upstream)


def _validate_operational_sidecar(
    ledger: pd.DataFrame, operational: pd.DataFrame
) -> None:
    if operational["query_id"].astype(str).duplicated().any():
        raise ValueError("Operational sidecar must be unique by query_id")
    exact_by_query = ledger.set_index("query_id")["human_ground_truth_siret"].to_dict()
    if set(operational["query_id"].astype(str)) != set(exact_by_query):
        raise ValueError("Operational sidecar must cover every exact labelled query")
    for row in operational.to_dict("records"):
        query_id = str(row["query_id"])
        exact = str(exact_by_query[query_id])
        acceptable = set(parse_operational(row["acceptable_sirets_operational_json"]))
        if acceptable != {exact}:
            raise ValueError(
                "v12 operational sidecar must be exact-only until sealed "
                f"same-site evidence exists: {query_id}"
            )


def _from_raw(
    layout: SourceLayout,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    _assert_no_model_or_retrieval_columns(layout.exact_path)
    available = set(_schema_names(layout.exact_path))
    required = {"query_id", "ground_truth_siret", "historical_ground_truth_siret"}
    if not required.issubset(available):
        raise ValueError(f"Raw source missing columns: {sorted(required - available)}")
    columns = [
        "query_id",
        "oof_fold" if "oof_fold" in available else "fold",
        "ground_truth_siret",
        "historical_ground_truth_siret",
        *_RAW_OPTIONAL_COLUMNS,
    ]
    raw = _scan_oof_only(layout.exact_path, columns=columns)
    usable, excluded = _base_label_rows(raw)
    ledger = _project(usable, LABEL_LEDGER_COLUMNS)
    operational, operational_audit = _operational_from_raw(usable)
    return ledger, operational, _project(excluded, EXCLUDED_COLUMNS), operational_audit


def _from_v11(
    layout: SourceLayout,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    assert layout.operational_path is not None and layout.excluded_path is not None
    _assert_no_model_or_retrieval_columns(layout.exact_path)
    exact = _scan_oof_only(layout.exact_path)
    required = {"query_id", "fold", "human_ground_truth_siret"}
    if not required.issubset(exact.columns):
        raise ValueError(f"v11 exact artifact missing columns: {sorted(required - set(exact))}")
    if exact["query_id"].isna().any() or exact["query_id"].astype(str).duplicated().any():
        raise ValueError("v11 exact artifact is not unique by query in folds 2/3/4")
    exact = exact.copy()
    exact["human_ground_truth_siret"] = exact["human_ground_truth_siret"].map(
        normalize_siret
    )
    if exact["human_ground_truth_siret"].eq("").any():
        raise ValueError("v11 exact artifact contains an invalid human SIRET")
    exact["human_ground_truth_siren"] = exact["human_ground_truth_siret"].str[:9]
    if "human_ground_truth_source" not in exact:
        raise ValueError("v11 exact artifact lacks resolved label provenance")
    ledger = _project(exact, LABEL_LEDGER_COLUMNS)

    operational_raw = _scan_oof_only(layout.operational_path)
    required_operational = {
        "query_id",
        "acceptable_sirets_operational_json",
    }
    if not required_operational.issubset(operational_raw.columns):
        raise ValueError("v11 operational artifact schema drifted")
    operational_raw["query_id"] = operational_raw["query_id"].astype(str)
    if operational_raw["query_id"].duplicated().any():
        raise ValueError("v11 operational artifact is not unique by query")
    upstream_operational = _project(operational_raw, OPERATIONAL_COLUMNS)
    if set(upstream_operational["query_id"]) != set(ledger["query_id"]):
        raise ValueError("v11 operational artifact does not cover the exact ledger")
    operational, operational_audit = _exact_only_operational(
        ledger,
        upstream_operational.set_index("query_id")[
            "acceptable_sirets_operational_json"
        ].to_dict(),
    )

    excluded_raw = _scan_oof_only(layout.excluded_path)
    if not excluded_raw.empty and "query_id" not in excluded_raw:
        raise ValueError("v11 missing-label artifact lacks query_id")
    if "missing_reason" in excluded_raw and not excluded_raw["missing_reason"].eq(
        ONLY_EXCLUSION
    ).all():
        raise ValueError("v11 missing-label artifact contains another exclusion")
    excluded = _project(excluded_raw, EXCLUDED_COLUMNS)
    if set(ledger["query_id"]).intersection(excluded["query_id"]):
        raise ValueError("A query is both labelled and excluded")
    return ledger, operational, excluded, operational_audit


def _fold_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["fold"].value_counts().to_dict() if not frame.empty else {}
    return {str(fold): int(counts.get(fold, 0)) for fold in OOF_FOLDS}


def _runtime_query_sample(ledger: pd.DataFrame) -> pd.DataFrame:
    if len(ledger) < RUNTIME_QUERY_SAMPLE_SIZE:
        raise ValueError(
            "v12 requires at least 512 all-human OOF labels for its sealed "
            "runtime sample"
        )
    sample = ledger[["query_id", "fold"]].copy()
    sample["query_id"] = sample["query_id"].astype(str)
    sample["fold"] = sample["fold"].astype(int)
    sample["_sample_key"] = sample["query_id"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    sample = (
        sample.sort_values(["_sample_key", "query_id"], kind="mergesort")
        .head(RUNTIME_QUERY_SAMPLE_SIZE)[["query_id", "fold"]]
        .reset_index(drop=True)
    )
    if len(sample) != RUNTIME_QUERY_SAMPLE_SIZE or sample["query_id"].duplicated().any():
        raise AssertionError("v12 runtime query sample is not exactly 512 unique rows")
    if set(sample["fold"]) != set(OOF_FOLDS):
        raise AssertionError("v12 runtime query sample must cover folds 2/3/4")
    return sample


def _validate_scanned_population(
    layout: SourceLayout, ledger: pd.DataFrame, excluded: pd.DataFrame
) -> None:
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    if layout.kind == "SEALED_RAW_ALL_HUMAN_DEVELOPMENT":
        expected = int((manifest.get("split_counts") or {}).get("crm_train", -1))
        if len(ledger) + len(excluded) != expected:
            raise ValueError("Raw all-human OOF population is an arbitrary subset")
        return
    if layout.kind == "SEALED_V11_DEVELOPMENT":
        population = manifest["population"]
        expected_usable = {
            str(fold): int(population["usable_by_fold"][str(fold)])
            for fold in OOF_FOLDS
        }
        expected_missing = {
            str(fold): int(population["missing_by_fold"][str(fold)])
            for fold in OOF_FOLDS
        }
        if _fold_counts(ledger) != expected_usable:
            raise ValueError("Sealed v11 exact OOF rows do not match population claims")
        if _fold_counts(excluded) != expected_missing:
            raise ValueError("Sealed v11 excluded OOF rows do not match population claims")
        return
    if layout.kind == "EXPLICIT_RAW_ALL_HUMAN_OOF":
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        if len(excluded) != 0 or len(ledger) != int(manifest["query_count"]):
            raise ValueError("Explicit raw source population does not match its claims")
        if _fold_counts(ledger) != manifest["counts_by_fold"]:
            raise ValueError("Explicit raw source fold counts do not match its claims")


def assert_siren_disjoint(ledger: pd.DataFrame) -> dict[str, Any]:
    overlaps: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    sirens_by_fold = {
        fold: set(
            ledger.loc[ledger["fold"].eq(fold), "human_ground_truth_siren"].astype(str)
        )
        for fold in OOF_FOLDS
    }
    for index, left in enumerate(OOF_FOLDS):
        for right in OOF_FOLDS[index + 1 :]:
            common = sorted(sirens_by_fold[left].intersection(sirens_by_fold[right]))
            key = f"{left}:{right}"
            overlaps[key] = len(common)
            if common:
                samples[key] = common[:10]
    if any(overlaps.values()):
        raise ValueError(
            "Ground-truth SIREN overlap across OOF folds: "
            + json.dumps({"counts": overlaps, "samples": samples}, sort_keys=True)
        )
    return {
        "ground_truth_siren_disjoint_across_folds": True,
        "pairwise_overlap_counts": overlaps,
        "unique_ground_truth_sirens_by_fold": {
            str(fold): len(sirens_by_fold[fold]) for fold in OOF_FOLDS
        },
    }


def _attach_official_catalog_presence(
    ledger: pd.DataFrame,
    establishments_path: Path | None,
    manifest_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach an exact GT-SIRET existence audit from a sealed official file.

    This is a label-side catalogue lookup only.  It neither generates nor
    ranks candidates and reads only the official ``siret`` column.
    """

    if establishments_path is None and manifest_path is None:
        output = ledger.copy()
        output["official_catalog_present"] = pd.Series(
            [pd.NA] * len(output), dtype="boolean"
        )
        return output, {
            "available": False,
            "reason": "NO_SEALED_OFFICIAL_ESTABLISHMENTS_SUPPLIED",
            "lookup_kind": "NONE",
        }
    if establishments_path is None or manifest_path is None:
        raise ValueError(
            "--official-establishments and --official-establishments-manifest "
            "must be supplied together"
        )
    establishments_path = Path(establishments_path)
    manifest_path = Path(manifest_path)
    if not establishments_path.is_file() or not manifest_path.is_file():
        raise ValueError("Sealed official establishments artifact is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        "sireto-siren-dossier-v3",
        "sireto-official-establishments-manifest-1",
    }:
        raise ValueError("Unsupported official establishments manifest schema")
    policy = manifest.get("policy") or {}
    if policy.get("crm_labels_present") is not False:
        raise ValueError("Official establishments source does not prove CRM labels absent")
    if policy.get("model_scores_present") is not False:
        raise ValueError("Official establishments source does not prove model scores absent")
    if schema_version == "sireto-official-establishments-manifest-1" and str(
        manifest.get("source") or ""
    ).upper() not in {"SIRENE", "SIRENE_STOCK_ETABLISSEMENT"}:
        raise ValueError("Explicit official establishments source is not SIRENE")
    declared = (manifest.get("files") or {}).get(establishments_path.name)
    if not isinstance(declared, Mapping):
        raise ValueError("Official manifest does not pin the establishments file")
    if declared.get("size_bytes") != establishments_path.stat().st_size:
        raise ValueError("Official establishments size mismatch")
    observed_sha = file_sha256(establishments_path)
    if declared.get("sha256") != observed_sha:
        raise ValueError("Official establishments hash mismatch")
    if "siret" not in _schema_names(establishments_path):
        raise ValueError("Official establishments artifact lacks a siret column")

    targets = set(ledger["human_ground_truth_siret"].astype(str))
    found: set[str] = set()
    scanner = ds.dataset(establishments_path, format="parquet").scanner(
        columns=["siret"],
        filter=ds.field("siret").isin(sorted(targets)),
        batch_size=131_072,
    )
    for batch in scanner.to_batches():
        found.update(
            normalize_siret(value)
            for value in batch.column(0).to_pylist()
            if normalize_siret(value)
        )
    output = ledger.copy()
    output["official_catalog_present"] = output[
        "human_ground_truth_siret"
    ].astype(str).isin(found).astype("boolean")
    return output, {
        "available": True,
        "lookup_kind": "EXACT_HUMAN_GT_SIRET_IN_SEALED_OFFICIAL_CATALOG",
        "manifest_schema_version": schema_version,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "establishments_path": str(establishments_path),
        "establishments_sha256": observed_sha,
        "establishments_size_bytes": establishments_path.stat().st_size,
        "target_siret_count": len(targets),
        "found_siret_count": len(found.intersection(targets)),
        "candidate_generation": False,
        "rank_or_score_read": False,
    }


def _schema_descriptor(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in pq.read_schema(path)
    ]


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_parquet(path, index=False, compression="zstd")
    return {
        "path": path.name,
        "rows": int(len(frame)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "schema": _schema_descriptor(path),
    }


def _validate_existing(destination: Path, build_id: str) -> None:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Existing v12 label-ledger destination is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("build_id") != build_id:
        raise ValueError("Existing v12 label-ledger identity mismatch")
    for name in (
        "label_ledger",
        "operational_sidecar",
        "excluded_sidecar",
        "runtime_query_sample",
    ):
        metadata = (manifest.get("artifacts") or {}).get(name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Existing v12 manifest lacks {name}")
        path = destination / str(metadata.get("path"))
        if not path.is_file() or file_sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Existing v12 artifact hash mismatch: {path.name}")


def build(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    contract_path = Path(args.contract)
    config_path = Path(args.config)
    output_root = Path(args.output_root)
    if not contract_path.is_file() or not config_path.is_file():
        raise ValueError("v12 contract and config must exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    layout = _resolve_source(source)

    if "RAW_ALL_HUMAN" in layout.kind:
        ledger, operational, excluded, operational_audit = _from_raw(layout)
    else:
        ledger, operational, excluded, operational_audit = _from_v11(layout)

    if set(ledger["fold"].unique()) != set(OOF_FOLDS):
        raise ValueError("Label ledger must contain every OOF fold 2/3/4")
    _validate_scanned_population(layout, ledger, excluded)
    if ledger["query_id"].duplicated().any():
        raise ValueError("Label ledger must be unique by query_id")
    if operational["query_id"].duplicated().any():
        raise ValueError("Operational sidecar must be unique by query_id")
    if set(operational["query_id"]) != set(ledger["query_id"]):
        raise ValueError("Operational sidecar must cover every labelled query")
    _validate_operational_sidecar(ledger, operational)
    if set(ledger["fold"]).intersection({0, 1}) or set(excluded["fold"]).intersection({0, 1}):
        raise AssertionError("fold 0/1 leaked into v12 label artifacts")
    official_establishments = getattr(args, "official_establishments", None)
    official_manifest = getattr(args, "official_establishments_manifest", None)
    ledger, official_lineage = _attach_official_catalog_presence(
        ledger,
        Path(official_establishments) if official_establishments is not None else None,
        Path(official_manifest) if official_manifest is not None else None,
    )
    disjoint_proof = assert_siren_disjoint(ledger)
    runtime_query_sample = _runtime_query_sample(ledger)

    source_hashes = {
        "manifest.json": file_sha256(layout.manifest_path),
        layout.exact_path.name: layout.exact_sha256,
    }
    if layout.operational_path is not None:
        source_hashes[layout.operational_path.name] = str(layout.operational_sha256)
    if layout.excluded_path is not None:
        source_hashes[layout.excluded_path.name] = str(layout.excluded_sha256)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "population": POPULATION,
        "only_exclusion": ONLY_EXCLUSION,
        "folds": list(OOF_FOLDS),
        "source_kind": layout.kind,
        "source_hashes": source_hashes,
        "contract_sha256": file_sha256(contract_path),
        "config_sha256": file_sha256(config_path),
        "builder_sha256": file_sha256(Path(__file__)),
        "label_precedence": [
            "ground_truth_siret",
            "historical_ground_truth_siret",
        ],
        "operational_equivalence_mode": OPERATIONAL_EQUIVALENCE_MODE,
        "runtime_query_sample_rule": RUNTIME_QUERY_SAMPLE_RULE,
        "positive_injection": False,
        "retrieval_outputs_read": False,
        "model_outputs_read": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "official_catalog": {
            "available": bool(official_lineage["available"]),
            "manifest_sha256": official_lineage.get("manifest_sha256"),
            "establishments_sha256": official_lineage.get("establishments_sha256"),
        },
    }
    build_id = canonical_json_sha256(identity)[:20]
    destination = output_root / build_id
    if destination.exists():
        _validate_existing(destination, build_id)
        return destination

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=output_root))
    try:
        artifacts = {
            "label_ledger": _write_parquet(ledger, temporary / "label_ledger.parquet"),
            "operational_sidecar": _write_parquet(
                operational, temporary / "operational_labels.parquet"
            ),
            "excluded_sidecar": _write_parquet(
                excluded, temporary / "excluded_missing_human_labels.parquet"
            ),
            "runtime_query_sample": _write_parquet(
                runtime_query_sample, temporary / "runtime_query_sample.parquet"
            ),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "population": POPULATION,
            "only_exclusion": ONLY_EXCLUSION,
            "positive_injection": False,
            "retrieval_outputs_read": False,
            "model_outputs_read": False,
            "operational_equivalence_mode": OPERATIONAL_EQUIVALENCE_MODE,
            "unverified_operational_alternative_count": operational_audit[
                "unverified_operational_alternative_count"
            ],
            "unverified_operational_alternative_count_by_fold": operational_audit[
                "unverified_operational_alternative_count_by_fold"
            ],
            "folds": list(OOF_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "query_count": int(len(ledger)),
            "excluded_count": int(len(excluded)),
            "counts_by_fold": _fold_counts(ledger),
            "excluded_counts_by_fold": _fold_counts(excluded),
            "human_ground_truth_source_counts": {
                str(key): int(value)
                for key, value in ledger["human_ground_truth_source"]
                .value_counts()
                .sort_index()
                .items()
            },
            "ground_truth_siren_disjoint_across_folds": True,
            "siren_disjoint_proof": disjoint_proof,
            "official_catalog_presence_available": bool(
                official_lineage["available"]
            ),
            "official_catalog_present_count": (
                int(ledger["official_catalog_present"].fillna(False).sum())
                if official_lineage["available"]
                else None
            ),
            "official_catalog_absent_count": (
                int(ledger["official_catalog_present"].eq(False).sum())
                if official_lineage["available"]
                else None
            ),
            "official_catalog_lineage": official_lineage,
            "label_ledger_path": artifacts["label_ledger"]["path"],
            "label_ledger_sha256": artifacts["label_ledger"]["sha256"],
            "operational_sidecar_path": artifacts["operational_sidecar"]["path"],
            "operational_sidecar_sha256": artifacts["operational_sidecar"]["sha256"],
            "excluded_sidecar_path": artifacts["excluded_sidecar"]["path"],
            "excluded_sidecar_sha256": artifacts["excluded_sidecar"]["sha256"],
            "runtime_query_sample_path": artifacts["runtime_query_sample"]["path"],
            "runtime_query_sample_sha256": artifacts["runtime_query_sample"]["sha256"],
            "runtime_query_sample_rows": RUNTIME_QUERY_SAMPLE_SIZE,
            "runtime_query_sample_rule": RUNTIME_QUERY_SAMPLE_RULE,
            "candidate_group_dependency": "NONE",
            "queries_with_zero_candidate_groups_remain_in_denominator": True,
            "source_scan": {
                "kind": layout.kind,
                "predicate": "fold IN (2,3,4)",
                "materialised_folds": list(OOF_FOLDS),
                "fold0_rows_materialised": False,
                "fold1_rows_materialised": False,
                "locked_fold1_artifacts_opened": False,
                "source_hashes": source_hashes,
            },
            "build_identity": identity,
            "artifacts": artifacts,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/retrieval_all_human_v12_contract.md"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/retrieval_all_human_v12.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-establishments", type=Path)
    parser.add_argument("--official-establishments-manifest", type=Path)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
