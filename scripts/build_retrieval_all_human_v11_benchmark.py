#!/usr/bin/env python3
"""Build the v11 all-human exact benchmark without selective qualification.

The builder is label-side only.  It never reads candidates, hits, ranks or
model scores.  Fold 1 is sealed separately and is absent from every
development artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import pandas as pd


SCHEMA_VERSION = "sireto-retrieval-all-human-v11-benchmark-1"
CONFIG_SCHEMA_VERSION = "sireto-retrieval-all-human-v11-config-1"
DEVELOPMENT_FOLDS = (0, 2, 3, 4)
TEST_FOLD = 1
ALL_FOLDS = (0, 1, 2, 3, 4)
SIRET_RE = re.compile(r"^[0-9]{14}$")
FORBIDDEN_INPUT_COLUMNS = (
    "retrieval_rank",
    "union_rank",
    "selected_rank",
    "ltr_score",
    "model_score",
    "exact_hit_at_100",
    "exact_oracle_hit",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_siret(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text if SIRET_RE.fullmatch(text) else ""


def parse_operational(value: Any) -> tuple[str, ...]:
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("acceptable_sirets_operational is not valid JSON") from exc
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("acceptable_sirets_operational must be a JSON/list collection")
    normalized = sorted({normalize_siret(item) for item in value} - {""})
    return tuple(normalized)


def _validate_source_manifest(source: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = source / "manifest.json"
    benchmark_path = source / "benchmark.parquet"
    if not manifest_path.is_file() or not benchmark_path.is_file():
        raise ValueError("Source must contain manifest.json and benchmark.parquet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or manifest.get("output_sha256") or {}
    expected = outputs.get("benchmark.parquet")
    if not expected:
        raise ValueError("Source manifest does not pin benchmark.parquet")
    observed = file_sha256(benchmark_path)
    if observed != expected:
        raise ValueError("Source benchmark hash mismatch")
    scope = (manifest.get("build_identity") or {}).get("population_scope")
    if scope not in (None, "all_human"):
        raise ValueError("Source population is not the all-human benchmark")
    if manifest.get("positive_injection") is True:
        raise ValueError("Source declares positive injection")
    return manifest, benchmark_path


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported v11 config schema")
    folds = config.get("folds") or {}
    if tuple(folds.get("train") or ()) != (2, 3, 4):
        raise ValueError("v11 train folds must be 2/3/4")
    if folds.get("development_diagnostic") != 0 or folds.get("test_locked") != 1:
        raise ValueError("v11 fold policy drift")
    labels = config.get("labels") or {}
    if tuple(labels.get("precedence") or ()) != (
        "ground_truth_siret",
        "historical_ground_truth_siret",
    ):
        raise ValueError("v11 exact-label precedence drift")
    if labels.get("qualification_fields_may_filter_primary_population") is not False:
        raise ValueError("Selective qualification is forbidden in v11")
    retrieval = config.get("retrieval") or {}
    expected_retrieval = {
        "raw_reference_cap": 10000,
        "maximum_internal_candidates": 2000,
        "maximum_final_candidates": 100,
        "parent_candidate_budget": 700,
        "relation_seed_limit": 32,
        "anchor_protection_slots": 20,
        "consensus_protection_slots": 20,
        "maximum_final_sites_per_siren": 3,
    }
    for key, expected in expected_retrieval.items():
        if retrieval.get(key) != expected:
            raise ValueError(f"v11 retrieval budget drift: {key}")
    rescue = retrieval.get("conditional_contrastive_rescue") or {}
    if (
        rescue.get("enabled") is not True
        or rescue.get("maximum_executions_per_query") != 1
        or rescue.get("unconditional_execution") is not False
    ):
        raise ValueError("v11 conditional contrastive rescue drift")
    development = config.get("development") or {}
    if (
        development.get("fold0_role") != "DIAGNOSTIC_ONLY_NOT_A_GATE"
        or development.get("fold0_required_before_fold1") is not False
        or development.get("fold0_may_select_variant_or_threshold") is not False
    ):
        raise ValueError("v11 fold0 diagnostic policy drift")
    gates = config.get("gates") or {}
    expected_gates = {
        "minimum_oof_oracle_recall_at_2000": 0.995,
        "minimum_oof_exact_recall_at_100": 0.992,
        "minimum_oof_exact_recall_at_100_wilson95_lower": 0.99,
        "minimum_each_outer_fold_exact_recall_at_100": 0.988,
        "maximum_warm_p95_ms": 1000,
        "maximum_warm_p99_ms": 2000,
    }
    for key, expected in expected_gates.items():
        if gates.get(key) != expected:
            raise ValueError(f"v11 gate drift: {key}")


def _required_columns(frame: pd.DataFrame) -> None:
    required = {
        "query_id",
        "oof_fold",
        "ground_truth_siret",
        "historical_ground_truth_siret",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source benchmark missing columns: {missing}")
    forbidden = sorted(set(FORBIDDEN_INPUT_COLUMNS) & set(frame.columns))
    if forbidden:
        raise ValueError(f"Retrieval/model output entered label builder: {forbidden}")


def _exact_projection(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "query_id",
        "crm_name",
        "crm_address",
        "crm_city",
        "postcode",
        "insee",
        "reference_date",
        "split",
        "oof_fold",
        "human_ground_truth_siret",
        "human_ground_truth_siren",
        "human_ground_truth_source",
        "ground_truth_state",
        "label_kind",
        "source_kind",
        "data_origin",
        "is_synthetic",
        "unseen_siren",
        "new_site_known_siren",
        "pool_size",
        "mega_base_pool",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns].sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _operational_projection(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    has_operational = "acceptable_sirets_operational" in frame.columns
    for row in frame.to_dict("records"):
        exact = str(row["human_ground_truth_siret"])
        acceptable = set(
            parse_operational(row.get("acceptable_sirets_operational"))
            if has_operational
            else ()
        )
        wrong_siren = sorted(value for value in acceptable if value[:9] != exact[:9])
        if wrong_siren:
            raise ValueError(
                f"Operational alternatives cross SIREN for {row['query_id']}: "
                f"{wrong_siren}"
            )
        acceptable.add(exact)
        ordered = sorted(acceptable)
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "oof_fold": int(row["oof_fold"]),
                "human_ground_truth_siret": exact,
                "acceptable_sirets_operational_json": json.dumps(
                    ordered, separators=(",", ":")
                ),
                "operational_equivalent_count": len(ordered) - 1,
            }
        )
    return pd.DataFrame(rows).sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _missing_projection(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "query_id",
        "oof_fold",
        "split",
        "label_kind",
        "qualification_v2",
        "qualification_v3",
        "source_kind",
        "data_origin",
    ]
    output = frame[[column for column in preferred if column in frame.columns]].copy()
    output["missing_reason"] = "NO_VALID_CURRENT_OR_HISTORICAL_HUMAN_SIRET"
    output["current_siret_present_but_invalid"] = frame[
        "ground_truth_siret"
    ].fillna("").astype(str).str.strip().ne("")
    output["historical_siret_present_but_invalid"] = frame[
        "historical_ground_truth_siret"
    ].fillna("").astype(str).str.strip().ne("")
    return output.sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_parquet(path, index=False)
    return {
        "path": path.name,
        "rows": int(len(frame)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _fold_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["oof_fold"].value_counts().to_dict() if not frame.empty else {}
    return {str(fold): int(counts.get(fold, 0)) for fold in ALL_FOLDS}


def _validate_existing(destination: Path, build_id: str) -> None:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Existing v11 destination is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("build_id") != build_id:
        raise ValueError("Existing v11 destination build id mismatch")
    for metadata in (manifest.get("outputs") or {}).values():
        path = destination / metadata["path"]
        if not path.is_file() or file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Existing v11 output hash mismatch: {path.name}")


def build(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    contract_path = Path(args.contract)
    config_path = Path(args.config)
    output_root = Path(args.output_root)
    _source_manifest, benchmark_path = _validate_source_manifest(source)
    if not contract_path.is_file() or not config_path.is_file():
        raise ValueError("v11 contract and config must exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)

    frame = pd.read_parquet(benchmark_path)
    _required_columns(frame)
    if frame["query_id"].isna().any() or frame["query_id"].astype(str).duplicated().any():
        raise ValueError("query_id must be non-null and unique")
    observed_folds = set(pd.to_numeric(frame["oof_fold"], errors="raise").astype(int))
    if not observed_folds.issubset(set(ALL_FOLDS)):
        raise ValueError(f"Unexpected folds: {sorted(observed_folds)}")
    frame = frame.copy()
    frame["oof_fold"] = frame["oof_fold"].astype(int)
    if "is_synthetic" in frame and frame["is_synthetic"].fillna(False).astype(bool).any():
        raise ValueError("Synthetic row entered all-human v11 benchmark")

    current = frame["ground_truth_siret"].map(normalize_siret)
    historical = frame["historical_ground_truth_siret"].map(normalize_siret)
    frame["human_ground_truth_siret"] = current.where(current.ne(""), historical)
    frame["human_ground_truth_source"] = "CURRENT"
    frame.loc[current.eq("") & historical.ne(""), "human_ground_truth_source"] = (
        "HISTORICAL_FALLBACK"
    )
    frame["human_ground_truth_siren"] = frame["human_ground_truth_siret"].str[:9]

    usable = frame[frame["human_ground_truth_siret"].ne("")].copy()
    missing = frame[frame["human_ground_truth_siret"].eq("")].copy()
    if len(usable) + len(missing) != len(frame):
        raise AssertionError("v11 population accounting failed")

    exact = _exact_projection(usable)
    operational = _operational_projection(usable)
    missing_list = _missing_projection(missing)
    development = exact[exact["oof_fold"].isin(DEVELOPMENT_FOLDS)].copy()
    test_locked = exact[exact["oof_fold"].eq(TEST_FOLD)].copy()
    operational_development = operational[
        operational["oof_fold"].isin(DEVELOPMENT_FOLDS)
    ].copy()
    operational_test_locked = operational[operational["oof_fold"].eq(TEST_FOLD)].copy()
    missing_development = missing_list[
        missing_list["oof_fold"].isin(DEVELOPMENT_FOLDS)
    ].copy()
    missing_test_locked = missing_list[missing_list["oof_fold"].eq(TEST_FOLD)].copy()

    if TEST_FOLD in set(development["oof_fold"]):
        raise AssertionError("fold 1 leaked into exact development benchmark")
    if TEST_FOLD in set(operational_development["oof_fold"]):
        raise AssertionError("fold 1 leaked into operational development labels")
    if TEST_FOLD in set(missing_development["oof_fold"]):
        raise AssertionError("fold 1 leaked into missing-label development list")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "source_benchmark_sha256": file_sha256(benchmark_path),
        "contract_sha256": file_sha256(contract_path),
        "config_sha256": file_sha256(config_path),
        "builder_sha256": file_sha256(Path(__file__)),
        "label_precedence": [
            "ground_truth_siret",
            "historical_ground_truth_siret",
        ],
        "only_exclusion": "NO_VALID_CURRENT_OR_HISTORICAL_HUMAN_SIRET",
        "development_folds": list(DEVELOPMENT_FOLDS),
        "test_locked_fold": TEST_FOLD,
        "positive_injection": False,
    }
    build_id = canonical_json_sha256(identity)[:16]
    destination = output_root / build_id
    if destination.exists():
        _validate_existing(destination, build_id)
        return destination

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=output_root))
    try:
        outputs = {
            "development": _write_parquet(development, temporary / "development.parquet"),
            "test_locked": _write_parquet(test_locked, temporary / "test_locked.parquet"),
            "operational_labels_development": _write_parquet(
                operational_development,
                temporary / "operational_labels_development.parquet",
            ),
            "operational_labels_test_locked": _write_parquet(
                operational_test_locked,
                temporary / "operational_labels_test_locked.parquet",
            ),
            "missing_human_labels_development": _write_parquet(
                missing_development,
                temporary / "missing_human_labels_development.parquet",
            ),
            "missing_human_labels_test_locked": _write_parquet(
                missing_test_locked,
                temporary / "missing_human_labels_test_locked.parquet",
            ),
        }
        outputs["development"]["folds"] = list(DEVELOPMENT_FOLDS)
        outputs["development"]["locked"] = False
        outputs["operational_labels_development"]["folds"] = list(DEVELOPMENT_FOLDS)
        outputs["operational_labels_development"]["locked"] = False
        outputs["missing_human_labels_development"]["folds"] = list(DEVELOPMENT_FOLDS)
        outputs["missing_human_labels_development"]["locked"] = False
        for name in (
            "test_locked",
            "operational_labels_test_locked",
            "missing_human_labels_test_locked",
        ):
            outputs[name]["folds"] = [TEST_FOLD]
            outputs[name]["locked"] = True

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "population": {
                "source_rows": int(len(frame)),
                "usable_human_labels": int(len(usable)),
                "excluded_missing_human_labels": int(len(missing)),
                "usable_by_fold": _fold_counts(usable),
                "missing_by_fold": _fold_counts(missing),
                "human_ground_truth_source_counts": {
                    str(key): int(value)
                    for key, value in usable["human_ground_truth_source"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            },
            "fold_lock": {
                "development_folds": list(DEVELOPMENT_FOLDS),
                "test_fold": TEST_FOLD,
                "fold1_locked": True,
                "fold1_opened": False,
                "development_contains_fold1": False,
                "test_consumption_requires_dedicated_authorization": True,
            },
            "metrics": {
                "primary": "EXACT_SIRET_ALL_USABLE_HUMAN_LABELS",
                "operational": "SECONDARY_LABEL_SIDE_ONLY",
                "selective_qualification_allowed": False,
            },
            "integrity": {
                "positive_injection": False,
                "retrieval_or_model_outputs_read": False,
                "synthetic_rows": 0,
                "operational_labels_physically_separate": True,
                "only_missing_labels_excluded": True,
            },
            "outputs": outputs,
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
        default=Path("docs/retrieval_all_human_v11_contract.md"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/retrieval_all_human_v11.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
