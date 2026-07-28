#!/usr/bin/env python3
"""Materialise V4.11 compact acceptor scenes from ranker-C OOF/dev scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v411_scene import (  # noqa: E402
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_BINARY_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
    V411_SCALED_FEATURE_NAMES,
    build_v411_compact_scene,
    validate_v411_scene_frame,
)
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.11-compact-acceptor-dataset-1"
EXPERIMENT_ID = "V411_COMPACT_ACCEPTOR_DATASET"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = Path("docs/v4_11_input_blind_aligned_stack_contract.md")
DEFAULT_TAXONOMY = Path("config/v4_9_site_function_taxonomy.json")
SCENE_SOURCE = Path("src/xgb_matcher/v411_scene.py")
SITE_FUNCTION_SOURCE = Path("src/xgb_matcher/v49_site_function.py")
PREDICTIONS_FILENAME = "predictions_ranker_c_oof_dev.parquet"
RANKER_PREDICTION_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "is_ground_truth",
    "ranker_score",
    "prediction_origin",
    "oof_fold",
    "ranker_rank",
]
LABEL_KINDS = {"MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED"}
RANKER_SCHEMA_VERSION = "sireto-v4.11-input-blind-ranker-c-development-1"
RANKER_RUN_ID = "V411_INPUT_BLIND_RANKER_C"
RANKER_REQUIRED_INVARIANTS = {
    "input_siret_or_siren_used_as_feature": False,
    "positive_injection": False,
    "acceptor_loaded_or_trained": False,
    "threshold_selected_or_applied": False,
    "fresh_challenge_opened": False,
    "final_test_opened": False,
    "five_oof_models": True,
    "one_full_fit_model": True,
    "two_repetitions_bit_exact": True,
}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _input_record(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(Path(path)),
        "size_bytes": int(Path(path).stat().st_size),
    }
    if row_count is not None:
        output["row_count"] = int(row_count)
    return output


def _declared_output_path(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    record = (manifest.get("outputs") or {}).get(filename)
    if not isinstance(record, Mapping):
        raise ValueError(f"STOP_INPUT_INTEGRITY: output not declared: {filename}")
    path = artifact_dir / filename
    if file_sha256(path) != str(record.get("sha256")):
        raise ValueError(f"STOP_INPUT_INTEGRITY: hash mismatch: {filename}")
    return path


def _validate_scene_implementation_inputs(manifest: Mapping[str, Any]) -> None:
    """Verify that every implementation used to derive scene features is pinned."""

    identity = manifest.get("build_identity") or {}
    inputs = manifest.get("inputs") or {}
    required = {
        "scene_source": ("scene_source_sha256", SCENE_SOURCE),
        "site_function_source": (
            "site_function_source_sha256",
            SITE_FUNCTION_SOURCE,
        ),
    }
    for input_name, (identity_name, relative_path) in required.items():
        record = inputs.get(input_name)
        if not isinstance(record, Mapping):
            raise ValueError(
                f"STOP_INPUT_INTEGRITY: scene implementation missing: {input_name}"
            )
        recorded_path = Path(str(record.get("path"))).resolve()
        expected_path = (REPO_ROOT / relative_path).resolve()
        expected_sha = str(record.get("sha256"))
        if recorded_path != expected_path:
            raise ValueError(
                f"STOP_INPUT_INTEGRITY: scene implementation path mismatch: "
                f"{input_name}"
            )
        if (
            not expected_sha
            or file_sha256(recorded_path) != expected_sha
            or str(identity.get(identity_name)) != expected_sha
        ):
            raise ValueError(
                f"STOP_INPUT_INTEGRITY: scene implementation drift: {input_name}"
            )


def _validate_ranker_artifact_link(
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
    ranker_artifact_dir: Path,
    ranker_manifest: Mapping[str, Any],
) -> None:
    """Require a passing, deterministic Ranker-C artifact built on this dataset."""

    if (
        ranker_manifest.get("schema_version") != RANKER_SCHEMA_VERSION
        or ranker_manifest.get("run_id") != RANKER_RUN_ID
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported ranker-C artifact")
    identity = ranker_manifest.get("build_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("STOP_INPUT_INTEGRITY: ranker-C build identity missing")
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if (
        str(ranker_manifest.get("build_id")) != expected_build_id
        or Path(ranker_artifact_dir).name != expected_build_id
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: ranker-C build identity mismatch")
    dataset_manifest_sha256 = file_sha256(Path(dataset_dir) / "manifest.json")
    if (
        str(ranker_manifest.get("dataset_manifest_sha256"))
        != dataset_manifest_sha256
        or str(ranker_manifest.get("dataset_build_id"))
        != str(dataset_manifest.get("build_id"))
    ):
        raise ValueError(
            "STOP_LEAKAGE: ranker-C was not trained on the acceptor retrieval dataset"
        )
    checks = ranker_manifest.get("checks")
    if (
        ranker_manifest.get("verdict") != "GO_RANKER_C"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: ranker-C did not pass frozen gates")
    invariants = ranker_manifest.get("invariants") or {}
    if any(
        invariants.get(name) is not expected
        for name, expected in RANKER_REQUIRED_INVARIANTS.items()
    ):
        raise ValueError("STOP_LEAKAGE: ranker-C invariants changed")
    outputs = ranker_manifest.get("outputs") or {}
    full_fit = outputs.get("ranker_c/full_fit.json")
    if not isinstance(full_fit, Mapping):
        raise ValueError("STOP_INPUT_INTEGRITY: ranker-C full-fit model missing")
    full_fit_path = Path(ranker_artifact_dir) / "ranker_c/full_fit.json"
    if file_sha256(full_fit_path) != str(full_fit.get("sha256")):
        raise ValueError("STOP_INPUT_INTEGRITY: ranker-C full-fit model hash mismatch")


def _dev_partition(component_id: str) -> str:
    digest = hashlib.sha256(
        f"v411-threshold:{component_id}".encode("utf-8")
    ).digest()
    return "threshold_dev" if digest[0] < 128 else "comparison_dev"


def _normalise_optional_identifier(value: Any, width: int) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    output = str(value).strip()
    if len(output) != width or not output.isdigit():
        raise ValueError("STOP_INPUT_INTEGRITY: invalid frozen identifier")
    return output


def _load_inputs(
    dataset_dir: Path,
    ranker_artifact_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    dataset_dir = Path(dataset_dir).resolve()
    ranker_artifact_dir = Path(ranker_artifact_dir).resolve()
    dataset_manifest_path = dataset_dir / "manifest.json"
    ranker_manifest_path = ranker_artifact_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    ranker_manifest = json.loads(ranker_manifest_path.read_text(encoding="utf-8"))
    if (
        dataset_manifest.get("schema_version")
        != "sireto-v4.11-input-blind-ranker-dataset-1"
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported V4.11 retrieval dataset")
    if (dataset_manifest.get("invariants") or {}).get("training_performed") is not False:
        raise ValueError("STOP_INPUT_INTEGRITY: retrieval dataset invariant changed")
    _validate_ranker_artifact_link(
        dataset_dir,
        dataset_manifest,
        ranker_artifact_dir,
        ranker_manifest,
    )
    queries_path = _declared_output_path(
        dataset_dir, dataset_manifest, "queries.parquet"
    )
    audit_path = _declared_output_path(
        dataset_dir, dataset_manifest, "query_audit.parquet"
    )
    labels_path = _declared_output_path(
        dataset_dir, dataset_manifest, "labels.parquet"
    )
    assignments_path = _declared_output_path(
        dataset_dir, dataset_manifest, "split_assignments.parquet"
    )
    candidates_path = _declared_output_path(
        dataset_dir, dataset_manifest, "candidates_sparse_top100.parquet"
    )
    predictions_path = _declared_output_path(
        ranker_artifact_dir, ranker_manifest, PREDICTIONS_FILENAME
    )
    return (
        dataset_manifest,
        ranker_manifest,
        pd.read_parquet(queries_path),
        pd.read_parquet(audit_path),
        pd.read_parquet(labels_path),
        pd.read_parquet(assignments_path),
        pd.read_parquet(candidates_path),
        predictions_path,
    )


def _validate_population(
    queries: pd.DataFrame,
    audit: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    enforce_canonical: bool,
) -> pd.DataFrame:
    for name, frame in (
        ("queries", queries),
        ("query_audit", audit),
        ("labels", labels),
        ("assignments", assignments),
    ):
        if "query_id" not in frame or frame["query_id"].astype(str).duplicated().any():
            raise ValueError(f"STOP_INPUT_INTEGRITY: invalid {name} query IDs")
    query_ids = set(queries["query_id"].astype(str))
    if any(
        set(frame["query_id"].astype(str)) != query_ids
        for frame in (audit, labels, assignments)
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: query populations differ")
    required_query = {"crm_name", "crm_address", "crm_city"}
    required_audit = {"input_siret_state", "source_segment"}
    required_labels = {
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
    }
    required_assignments = {
        "siren_component_id",
        "split",
        "oof_fold",
    }
    for observed, required, name in (
        (set(queries), required_query, "queries"),
        (set(audit), required_audit, "query_audit"),
        (set(labels), required_labels, "labels"),
        (set(assignments), required_assignments, "assignments"),
    ):
        if not required.issubset(observed):
            raise ValueError(
                f"STOP_INPUT_INTEGRITY: {name} missing {sorted(required-observed)}"
            )
    if not set(labels["label_kind"].astype(str)).issubset(LABEL_KINDS):
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported label kind")
    if set(assignments["split"].astype(str)) != {"fit", "dev"}:
        raise ValueError("STOP_INPUT_INTEGRITY: split registry changed")
    folds = set(assignments["oof_fold"].astype(int))
    if not folds or not folds.issubset(set(range(5))):
        raise ValueError("STOP_INPUT_INTEGRITY: frozen OOF folds changed")
    merged = (
        queries.merge(audit, on="query_id", validate="one_to_one")
        .merge(labels, on="query_id", validate="one_to_one")
        .merge(assignments, on="query_id", validate="one_to_one")
    )
    dev = merged["split"].eq("dev")
    merged["dev_partition"] = ""
    merged.loc[dev, "dev_partition"] = merged.loc[
        dev, "siren_component_id"
    ].astype(str).map(_dev_partition)
    leakage = (
        merged.loc[dev]
        .groupby("siren_component_id")["dev_partition"]
        .nunique()
        .gt(1)
        .any()
    )
    if leakage:
        raise ValueError("STOP_LEAKAGE: component crosses dev partitions")
    component_split_counts = merged.groupby("siren_component_id")["split"].nunique()
    if component_split_counts.gt(1).any():
        raise ValueError("STOP_LEAKAGE: component crosses fit/dev")
    fit = merged["split"].eq("fit")
    component_fold_counts = (
        merged.loc[fit].groupby("siren_component_id")["oof_fold"].nunique()
    )
    if component_fold_counts.gt(1).any():
        raise ValueError("STOP_LEAKAGE: component crosses OOF folds")
    if enforce_canonical:
        expected = {
            ("threshold_dev", "MATCH_EXACT"): 583,
            ("threshold_dev", "AMBIGUOUS"): 127,
            ("comparison_dev", "MATCH_EXACT"): 634,
            ("comparison_dev", "AMBIGUOUS"): 112,
        }
        observed = (
            merged.loc[dev]
            .groupby(["dev_partition", "label_kind"])
            .size()
            .to_dict()
        )
        fit_counts = (
            merged.loc[fit, "label_kind"].astype(str).value_counts().to_dict()
        )
        component_counts = (
            merged.loc[dev]
            .groupby("dev_partition")["siren_component_id"]
            .nunique()
            .to_dict()
        )
        if (
            observed != expected
            or fit_counts != {"MATCH_EXACT": 4666, "AMBIGUOUS": 881}
            or component_counts
            != {"threshold_dev": 637, "comparison_dev": 652}
            or int(fit.sum()) != 5547
            or len(merged) != 7003
            or folds != set(range(5))
        ):
            raise ValueError(
                "STOP_INPUT_INTEGRITY: canonical populations changed: "
                f"dev={observed}, fit={fit_counts}, components={component_counts}"
            )
    return merged


def _join_ranker_scores(
    candidates: pd.DataFrame,
    predictions_path: Path,
    population: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    if list(predictions.columns) != RANKER_PREDICTION_COLUMNS:
        raise ValueError("STOP_INPUT_INTEGRITY: ranker prediction schema changed")
    keys = ["query_id", "candidate_siret"]
    if candidates.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError("STOP_INPUT_INTEGRITY: duplicate candidate prediction")
    if len(candidates) != len(predictions):
        raise ValueError("STOP_INPUT_INTEGRITY: ranker prediction cardinality changed")
    candidate_keys = set(map(tuple, candidates[keys].astype(str).to_numpy()))
    prediction_keys = set(map(tuple, predictions[keys].astype(str).to_numpy()))
    if candidate_keys != prediction_keys:
        raise ValueError("STOP_INPUT_INTEGRITY: ranker did not score exact pools")
    score_values = predictions["ranker_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(score_values).all():
        raise ValueError("STOP_INPUT_INTEGRITY: non-finite ranker score")
    joined = candidates.merge(
        predictions[
            [
                "query_id",
                "candidate_siret",
                "candidate_siren",
                "retrieval_rank",
                "is_ground_truth",
                "ranker_score",
                "prediction_origin",
                "oof_fold",
            ]
        ],
        on=keys,
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    for name in ("candidate_siren", "retrieval_rank", "is_ground_truth"):
        left = joined[name].astype(str)
        right = joined[f"{name}_prediction"].astype(str)
        if not left.equals(right):
            raise ValueError(f"STOP_INPUT_INTEGRITY: ranker {name} drift")
        joined = joined.drop(columns=f"{name}_prediction")
    assignment = population.set_index("query_id")[["split", "oof_fold"]]
    joined = joined.join(assignment, on="query_id", rsuffix="_assignment")
    fit = joined["split"].eq("fit")
    if not joined.loc[fit, "prediction_origin"].eq("ranker_c_oof").all():
        raise ValueError("STOP_LEAKAGE: fit scenes are not ranker-C OOF")
    if not joined.loc[~fit, "prediction_origin"].eq("ranker_c_dev").all():
        raise ValueError("STOP_LEAKAGE: dev scenes are not out-of-sample")
    if not joined.loc[fit, "oof_fold"].astype(int).equals(
        joined.loc[fit, "oof_fold_assignment"].astype(int)
    ):
        raise ValueError("STOP_LEAKAGE: ranker OOF fold mismatch")
    return joined.drop(columns=["split", "oof_fold_assignment"])


def build_scene_frame(
    population: pd.DataFrame,
    candidates: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
) -> pd.DataFrame:
    """Build one train/serve-identical compact scene per query."""

    grouped = {
        str(query_id): frame
        for query_id, frame in candidates.groupby("query_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for query in population.sort_values("query_id", kind="mergesort").to_dict(
        "records"
    ):
        query_id = str(query["query_id"])
        pool = grouped.get(query_id, candidates.iloc[0:0])
        scene = build_v411_compact_scene(query, pool, taxonomy)
        truth_siret = _normalise_optional_identifier(
            query.get("ground_truth_siret"), 14
        )
        truth_siren = _normalise_optional_identifier(
            query.get("ground_truth_siren"), 9
        )
        predicted_siret = scene.pop("predicted_siret")
        predicted_siren = scene.pop("predicted_siren")
        label_kind = str(query["label_kind"])
        target: int | None
        if label_kind == "UNRESOLVED":
            target = None
        else:
            target = int(
                label_kind == "MATCH_EXACT"
                and predicted_siret is not None
                and predicted_siret == truth_siret
            )
        rows.append(
            {
                "query_id": query_id,
                "crm_record_id": str(query["crm_record_id"]),
                "split": str(query["split"]),
                "dev_partition": str(query["dev_partition"]),
                "oof_fold": int(query["oof_fold"]),
                "siren_component_id": str(query["siren_component_id"]),
                "label_kind": label_kind,
                "ground_truth_siret": truth_siret,
                "ground_truth_siren": truth_siren,
                "predicted_siret": predicted_siret,
                "predicted_siren": predicted_siren,
                "acceptor_target": target,
                "ranker_prediction_is_out_of_sample": True,
                "prediction_origin": (
                    "ranker_c_oof" if query["split"] == "fit" else "ranker_c_dev"
                ),
                "input_siret_state": str(query["input_siret_state"]),
                "source_segment": str(query["source_segment"]),
                **scene,
            }
        )
    output = pd.DataFrame(rows)
    validate_v411_scene_frame(output)
    return output


def validate_scene_artifact(
    artifact_dir: Path,
    *,
    enforce_canonical: bool = True,
) -> None:
    artifact_dir = Path(artifact_dir).resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported scene artifact")
    identity = manifest.get("build_identity") or {}
    expected_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != expected_id or artifact_dir.name != expected_id:
        raise ValueError("STOP_INPUT_INTEGRITY: scene build identity mismatch")
    _validate_scene_implementation_inputs(manifest)
    scenes_path = _declared_output_path(
        artifact_dir, manifest, "acceptor_scenes.parquet"
    )
    scenes = pd.read_parquet(scenes_path)
    validate_v411_scene_frame(scenes)
    if scenes["query_id"].astype(str).duplicated().any():
        raise ValueError("STOP_INPUT_INTEGRITY: duplicate acceptor scene")
    if enforce_canonical and len(scenes) != 7003:
        raise ValueError("STOP_INPUT_INTEGRITY: canonical scene count changed")
    invariants = manifest.get("invariants") or {}
    required = {
        "ranker_fit_predictions_out_of_fold": True,
        "ranker_dev_predictions_out_of_sample": True,
        "consumed_hard_opened": False,
        "random_or_locked_opened": False,
        "acceptor_trained": False,
        "dev_partition_label_blind": True,
    }
    if any(invariants.get(key) != value for key, value in required.items()):
        raise ValueError("STOP_INPUT_INTEGRITY: scene invariant changed")


def build_dataset(
    *,
    dataset_dir: Path,
    ranker_artifact_dir: Path,
    taxonomy_path: Path,
    contract_path: Path,
    output_root: Path,
    enforce_canonical: bool = True,
) -> Path:
    (
        dataset_manifest,
        ranker_manifest,
        queries,
        audit,
        labels,
        assignments,
        candidates,
        predictions_path,
    ) = _load_inputs(dataset_dir, ranker_artifact_dir)
    population = _validate_population(
        queries,
        audit,
        labels,
        assignments,
        enforce_canonical=enforce_canonical,
    )
    scored_candidates = _join_ranker_scores(
        candidates, predictions_path, population
    )
    taxonomy = SiteFunctionTaxonomy.load(taxonomy_path)
    scenes = build_scene_frame(population, scored_candidates, taxonomy)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "retrieval_dataset_manifest_sha256": file_sha256(
            Path(dataset_dir) / "manifest.json"
        ),
        "ranker_artifact_manifest_sha256": file_sha256(
            Path(ranker_artifact_dir) / "manifest.json"
        ),
        "ranker_predictions_sha256": file_sha256(predictions_path),
        "taxonomy_sha256": file_sha256(taxonomy_path),
        "contract_sha256": file_sha256(contract_path),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "scene_source_sha256": file_sha256(REPO_ROOT / SCENE_SOURCE),
        "site_function_source_sha256": file_sha256(
            REPO_ROOT / SITE_FUNCTION_SOURCE
        ),
        "feature_order": V411_ACCEPTOR_FEATURE_NAMES,
        "scaled_features": V411_SCALED_FEATURE_NAMES,
        "binary_features": V411_BINARY_FEATURE_NAMES,
        "monotonic_constraints": V411_MONOTONIC_CONSTRAINTS,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.11 scene dataset exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    try:
        scenes_path = staging / "acceptor_scenes.parquet"
        scenes.to_parquet(scenes_path, index=False)
        split_counts = (
            scenes.groupby(["split", "dev_partition", "label_kind"], dropna=False)
            .size()
            .rename("row_count")
            .reset_index()
            .to_dict("records")
        )
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "retrieval_dataset_manifest": _input_record(
                    Path(dataset_dir) / "manifest.json"
                ),
                "ranker_artifact_manifest": _input_record(
                    Path(ranker_artifact_dir) / "manifest.json"
                ),
                "ranker_predictions": _input_record(
                    predictions_path, row_count=len(scored_candidates)
                ),
                "taxonomy": _input_record(taxonomy_path),
                "contract": _input_record(contract_path),
                "builder_source": _input_record(Path(__file__).resolve()),
                "scene_source": _input_record(REPO_ROOT / SCENE_SOURCE),
                "site_function_source": _input_record(
                    REPO_ROOT / SITE_FUNCTION_SOURCE
                ),
            },
            "upstream": {
                "retrieval_build_id": dataset_manifest.get("build_id"),
                "ranker_build_id": ranker_manifest.get("build_id"),
            },
            "outputs": {
                "acceptor_scenes.parquet": {
                    "sha256": file_sha256(scenes_path),
                    "size_bytes": int(scenes_path.stat().st_size),
                    "row_count": int(len(scenes)),
                }
            },
            "volumes": {
                "scenes": len(scenes),
                "candidates_scored": len(scored_candidates),
                "split_label_counts": split_counts,
            },
            "invariants": {
                "ranker_fit_predictions_out_of_fold": True,
                "ranker_dev_predictions_out_of_sample": True,
                "consumed_hard_opened": False,
                "random_or_locked_opened": False,
                "acceptor_trained": False,
                "dev_partition_label_blind": True,
                "input_identifiers_in_model_features": False,
                "feature_count": 80,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_scene_artifact(target, enforce_canonical=enforce_canonical)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--ranker-artifact", type=Path)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--allow-noncanonical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact:
        validate_scene_artifact(
            args.validate_artifact,
            enforce_canonical=not args.allow_noncanonical,
        )
        print(args.validate_artifact)
        return
    missing = [
        name
        for name in ("dataset", "ranker_artifact", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    target = build_dataset(
        dataset_dir=args.dataset,
        ranker_artifact_dir=args.ranker_artifact,
        taxonomy_path=args.taxonomy,
        contract_path=args.contract,
        output_root=args.output_root,
        enforce_canonical=not args.allow_noncanonical,
    )
    print(target)


if __name__ == "__main__":
    main()
