#!/usr/bin/env python3
"""Open the V4.8 random holdout once, after the winner has been frozen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import run_v48_acceptor_development as development_runner  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402

FROZEN_THRESHOLD = development_runner.FROZEN_THRESHOLD
decision_metrics = development_runner.decision_metrics
model_scores = development_runner.model_scores


SCHEMA_VERSION = "sireto-v4.8-random-holdout-1"
WINNER_THRESHOLD = 0.3617231974526733
CANONICAL_OUTPUT_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_random_holdout"
)
CANONICAL_LEDGER_PATH = CANONICAL_OUTPUT_ROOT / "OPENING_LEDGER.json"
EXPECTED_HASHES = {
    "contract": "b1f6fba65d15b7fe88e5bea493b0a070ef801b3f260e6576cb140da3ec07bee2",
    "partition_manifest": "f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33",
    "partition_assignments": "f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9",
    "current_labels": "e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2",
    "development_manifest": "a232ff17fd708321a3129f9411626cdfea5c5d46f8c69a9776ace474f23888d4",
    "development_report": "e8848c7f4c8ecbdb532194519d8c90eba39011813c139a09c9266f70c568b2e2",
    "winner_model": "2423033ef5e003112481fb58926611dbfbaf71b8562aea848545c5ab098e487c",
    "winner_metadata": "41b84f05fe846db9362b1eff5f362b075bec08aee3af1bd1c5ee553d5d56abfc",
    "winner_freeze": "5d7344b2e4b2fa256f05e75420a5c16edaf52a530f6e9486000aeaec74c8bcbc",
    "frozen_model": "16283b8aba5ed135846a74e9040c79e9f863f7e2bd658ca642ad444174b9a3fa",
    "frozen_metadata": "73199451b2de6ae383c9c0c58b10ab9c7393994a4efdec45f9c8e1e9f150691c",
    "development_runner": "1fac5f3426a53fb450c6d7b2a532563634aadd9378561676bfa5ac932537b1f1",
}


def _json_dump_durable(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    with path.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def create_exclusive_ledger(path: Path, payload: Any) -> None:
    """Create the one-shot global receipt atomically; an existing file forbids retry."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def random_gate(
    *,
    winner_predictions: pd.DataFrame,
    frozen_predictions: pd.DataFrame,
) -> dict[str, Any]:
    winner = winner_predictions.copy()
    frozen = frozen_predictions.copy()
    if (
        winner["query_id"].astype(str).duplicated().any()
        or frozen["query_id"].astype(str).duplicated().any()
        or set(winner["query_id"].astype(str)) != set(frozen["query_id"].astype(str))
        or len(winner) != 52
        or len(frozen) != 52
    ):
        raise ValueError("STOP_RANDOM_INTEGRITY: paired random IDs are invalid")
    negative = winner["adjudication_label"].isin({"TOP1_WRONG", "AMBIGUOUS"})
    correct = winner["adjudication_label"].eq("TOP1_CORRECT")
    frozen_correct = frozen["adjudication_label"].eq("TOP1_CORRECT")
    winner_negative_auto = int((negative & winner["auto"]).sum())
    winner_correct_auto = int((correct & winner["auto"]).sum())
    frozen_correct_auto = int((frozen_correct & frozen["auto"]).sum())
    winner_error_auto = int(
        (~winner["acceptor_target"].astype(bool) & winner["auto"]).sum()
    )
    checks = {
        "zero_negative_auto": winner_negative_auto == 0,
        "zero_error_auto": winner_error_auto == 0,
        "at_least_20_correct_auto": winner_correct_auto >= 20,
        "at_most_one_correct_auto_below_frozen": (
            winner_correct_auto >= frozen_correct_auto - 1
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "winner_negative_auto": winner_negative_auto,
        "winner_error_auto": winner_error_auto,
        "winner_correct_auto": winner_correct_auto,
        "frozen_correct_auto": frozen_correct_auto,
        "correct_auto_delta": winner_correct_auto - frozen_correct_auto,
    }


def _validate_preopening(
    *,
    paths: dict[str, Path],
    hashes: dict[str, str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    mismatches = {
        name: (EXPECTED_HASHES[name], actual)
        for name, actual in hashes.items()
        if actual != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"V4.8 random input mismatch: {mismatches}")
    partition_manifest = json.loads(
        paths["partition_manifest"].read_text(encoding="utf-8")
    )
    if partition_manifest.get("invariants", {}).get("random_targets_exposed") is not False:
        raise ValueError("V4.8 random targets were exposed by the partition manifest")
    assignments = pd.read_parquet(paths["partition_assignments"])
    assignments["query_id"] = assignments["query_id"].astype(str)
    random_assignments = assignments[
        assignments["population"].eq("current")
        & assignments["role"].eq("random_sealed")
    ].copy()
    if (
        len(random_assignments) != 57
        or random_assignments["acceptor_target"].notna().any()
        or random_assignments["label_visible"].astype(bool).any()
        or not random_assignments["partition"].eq("random_sealed").all()
    ):
        raise ValueError("V4.8 random partition integrity failed before opening")
    winner_metadata = json.loads(
        paths["winner_metadata"].read_text(encoding="utf-8")
    )
    winner_freeze = json.loads(paths["winner_freeze"].read_text(encoding="utf-8"))
    development_report = json.loads(
        paths["development_report"].read_text(encoding="utf-8")
    )
    development_manifest = json.loads(
        paths["development_manifest"].read_text(encoding="utf-8")
    )
    expected_dependencies = development_manifest.get("dependencies", {})
    current_dependencies = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if current_dependencies != expected_dependencies:
        raise ValueError(
            f"V4.8 random runtime drift: {current_dependencies} "
            f"!= {expected_dependencies}"
        )
    development_invariants = development_manifest.get("invariants", {})
    if (
        development_invariants.get("random_rows_scored") != 0
        or development_invariants.get("random_targets_read_from_partition") is not False
        or development_invariants.get("test_opened") is not False
    ):
        raise ValueError("V4.8 development manifest does not preserve the holdout")
    if (
        winner_metadata.get("variant") != "HARD_W1"
        or float(winner_metadata.get("threshold")) != WINNER_THRESHOLD
        or winner_freeze.get("variant") != "HARD_W1"
        or float(winner_freeze.get("threshold")) != WINNER_THRESHOLD
        or winner_freeze.get("random_opened") is not False
        or development_report.get("development_verdict") != "GO_RANDOM_OPEN_V48"
        or development_report.get("winner") != "HARD_W1"
    ):
        raise ValueError("V4.8 winner freeze does not authorize random opening")
    feature_order = [str(value) for value in winner_metadata.get("feature_order", [])]
    if len(feature_order) != 80 or len(set(feature_order)) != 80:
        raise ValueError("V4.8 random opening requires 80 unique features")
    return random_assignments, feature_order, winner_metadata


def open_random(
    *,
    contract_path: Path,
    partition_manifest_path: Path,
    partition_assignments_path: Path,
    current_labels_path: Path,
    development_manifest_path: Path,
    development_report_path: Path,
    winner_model_path: Path,
    winner_metadata_path: Path,
    winner_freeze_path: Path,
    frozen_model_path: Path,
    frozen_metadata_path: Path,
    output_root: Path,
) -> Path:
    if Path(output_root).resolve() != CANONICAL_OUTPUT_ROOT:
        raise ValueError("V4.8 random output root is canonical and cannot be changed")
    paths = {
        "contract": Path(contract_path).resolve(),
        "partition_manifest": Path(partition_manifest_path).resolve(),
        "partition_assignments": Path(partition_assignments_path).resolve(),
        "current_labels": Path(current_labels_path).resolve(),
        "development_manifest": Path(development_manifest_path).resolve(),
        "development_report": Path(development_report_path).resolve(),
        "winner_model": Path(winner_model_path).resolve(),
        "winner_metadata": Path(winner_metadata_path).resolve(),
        "winner_freeze": Path(winner_freeze_path).resolve(),
        "frozen_model": Path(frozen_model_path).resolve(),
        "frozen_metadata": Path(frozen_metadata_path).resolve(),
        "development_runner": Path(development_runner.__file__).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    random_assignments, feature_order, winner_metadata = _validate_preopening(
        paths=paths,
        hashes=hashes,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": hashes,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "winner_variant": "HARD_W1",
        "winner_threshold": WINNER_THRESHOLD,
        "random_query_ids_sha256": hashlib.sha256(
            "\n".join(sorted(random_assignments["query_id"])).encode("utf-8")
        ).hexdigest(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    target = CANONICAL_OUTPUT_ROOT / build_id
    if CANONICAL_LEDGER_PATH.exists():
        raise FileExistsError(
            f"V4.8 random holdout was already opened: {CANONICAL_LEDGER_PATH}"
        )
    CANONICAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"V4.8 random receipt already exists: {target}")
    target.mkdir(exist_ok=True)
    marker_path = target / "random_opening_marker.json"
    status_path = target / "random_opening_status.json"
    ledger = {
        **identity,
        "build_id": build_id,
        "target": str(target),
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "opening_status": "OPENED_ONCE_GLOBAL",
        "retry_forbidden": True,
        "test_opened": False,
    }

    ledger_acquired = False
    try:
        create_exclusive_ledger(CANONICAL_LEDGER_PATH, ledger)
        ledger_acquired = True
        marker = {
            **identity,
            "build_id": build_id,
            "global_ledger_path": str(CANONICAL_LEDGER_PATH),
            "global_ledger_sha256": file_sha256(CANONICAL_LEDGER_PATH),
            "opened_at": ledger["opened_at"],
            "opening_status": "OPENED_ONCE",
            "written_before_semantic_random_read": True,
            "written_before_model_scoring": True,
            "test_opened": False,
        }
        _json_dump_durable(marker_path, marker)
        status = {
            "opening_status": "OPENED_ONCE_IN_PROGRESS",
            "semantic_random_read_started": True,
            "semantic_random_read_completed": False,
            "model_scoring_started": False,
            "model_scoring_completed": False,
            "test_opened": False,
        }
        _json_dump_durable(status_path, status)
        random_ids = sorted(random_assignments["query_id"].astype(str))
        columns = [
            "query_id",
            "audit_case_id",
            "sampling_stratum",
            "current_adjudication_label",
            "current_evidence_validated",
            "current_training_eligible",
            "current_acceptor_target",
            "current_label_origin",
            *feature_order,
        ]
        current = pd.read_parquet(
            paths["current_labels"],
            columns=columns,
            filters=[("query_id", "in", random_ids)],
        )
        current["query_id"] = current["query_id"].astype(str)
        if set(current["query_id"]) != set(random_ids) or len(current) != 57:
            raise ValueError("STOP_RANDOM_INTEGRITY: random filtered read is incomplete")
        if not current["sampling_stratum"].eq("RANDOM_POPULATION").all():
            raise ValueError("STOP_RANDOM_INTEGRITY: a non-random row was opened")
        reliable = current[
            current["current_evidence_validated"].astype(bool)
            & current["current_training_eligible"].astype(bool)
            & current["current_acceptor_target"].notna()
        ].copy()
        counts = reliable["current_adjudication_label"].value_counts().to_dict()
        expected_counts = {"TOP1_CORRECT": 46, "TOP1_WRONG": 5, "AMBIGUOUS": 1}
        if len(reliable) != 52 or counts != expected_counts:
            raise ValueError(
                f"STOP_RANDOM_INTEGRITY: reliable random labels={counts}"
            )
        targets = reliable["current_acceptor_target"].astype(int)
        expected_targets = reliable["current_adjudication_label"].map(
            {"TOP1_CORRECT": 1, "TOP1_WRONG": 0, "AMBIGUOUS": 0}
        )
        if not targets.eq(expected_targets).all():
            raise ValueError("STOP_RANDOM_INTEGRITY: random targets are incoherent")
        matrix = reliable[feature_order].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError("STOP_RANDOM_INTEGRITY: non-finite random feature")

        status["semantic_random_read_completed"] = True
        status["model_scoring_started"] = True
        _json_dump_durable(status_path, status)
        winner_model = joblib.load(paths["winner_model"])
        frozen_model = joblib.load(paths["frozen_model"])
        winner_scores = model_scores(
            winner_model, reliable, feature_order=feature_order
        )
        frozen_scores = model_scores(
            frozen_model, reliable, feature_order=feature_order
        )
        status["model_scoring_completed"] = True
        _json_dump_durable(status_path, status)

        base_columns = [
            "query_id",
            "audit_case_id",
            "current_adjudication_label",
            "current_acceptor_target",
            "current_label_origin",
        ]
        winner_predictions = reliable[base_columns].rename(
            columns={
                "current_adjudication_label": "adjudication_label",
                "current_acceptor_target": "acceptor_target",
            }
        )
        winner_predictions["variant"] = "HARD_W1"
        winner_predictions["score"] = winner_scores
        winner_predictions["threshold"] = WINNER_THRESHOLD
        winner_predictions["auto"] = winner_scores >= WINNER_THRESHOLD
        frozen_predictions = reliable[base_columns].rename(
            columns={
                "current_adjudication_label": "adjudication_label",
                "current_acceptor_target": "acceptor_target",
            }
        )
        frozen_predictions["variant"] = "BASE_FROZEN"
        frozen_predictions["score"] = frozen_scores
        frozen_predictions["threshold"] = FROZEN_THRESHOLD
        frozen_predictions["auto"] = frozen_scores >= FROZEN_THRESHOLD
        predictions = pd.concat(
            [frozen_predictions, winner_predictions],
            ignore_index=True,
        )
        gate = random_gate(
            winner_predictions=winner_predictions,
            frozen_predictions=frozen_predictions,
        )
        metrics: dict[str, Any] = {}
        for variant, frame in predictions.groupby("variant", sort=False):
            metrics[str(variant)] = decision_metrics(
                frame["score"].to_numpy(),
                frame["acceptor_target"].astype(int).to_numpy(),
                float(frame["threshold"].iloc[0]),
                ambiguous=frame["adjudication_label"].eq("AMBIGUOUS").to_numpy(),
            )
        verdict = "GO_FRESH_SHADOW_V48" if gate["passed"] else "STOP_RETRAIN"
        report = {
            "schema_version": SCHEMA_VERSION,
            "verdict": verdict,
            "gate": gate,
            "metrics": metrics,
            "label_counts": expected_counts,
            "reliable_random_count": len(reliable),
            "unresolved_random_count": int(len(current) - len(reliable)),
            "winner_variant": winner_metadata["variant"],
            "winner_threshold": WINNER_THRESHOLD,
            "limitations": [
                "Only six reliable random negatives are available.",
                "Zero observed error is not a 99.8% certification.",
                "Random class counts were known in aggregate before opening.",
            ],
        }
        predictions_path = target / "random_predictions.parquet"
        report_path = target / "random_report.json"
        predictions.to_parquet(predictions_path, index=False)
        _json_dump_durable(report_path, report)
        status["completed_at"] = datetime.now(timezone.utc).isoformat()
        status["opening_status"] = "OPENED_ONCE_COMPLETED"
        status["verdict"] = verdict
        _json_dump_durable(status_path, status)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: str(path) for name, path in paths.items()},
            "outputs": {
                "OPENING_LEDGER.json": file_sha256(CANONICAL_LEDGER_PATH),
                marker_path.name: file_sha256(marker_path),
                status_path.name: file_sha256(status_path),
                predictions_path.name: file_sha256(predictions_path),
                report_path.name: file_sha256(report_path),
            },
            "verdict": verdict,
            "invariants": {
                "opening_count": 1,
                "winner_modified": False,
                "threshold_modified": False,
                "random_rows_scored": 52,
                "unresolved_random_rows_scored": 0,
                "retrieval_trained": False,
                "ranker_trained": False,
                "test_opened": False,
            },
        }
        _json_dump_durable(target / "manifest.json", manifest)
    except Exception as error:
        if not ledger_acquired:
            raise
        failed_status = (
            dict(status)
            if "status" in locals()
            else {
                "semantic_random_read_started": False,
                "model_scoring_started": False,
                "test_opened": False,
            }
        )
        failed_status["failed_at"] = datetime.now(timezone.utc).isoformat()
        failed_status["opening_status"] = "OPENED_ONCE_FAILED"
        failed_status["verdict"] = "STOP_RANDOM_INTEGRITY"
        _json_dump_durable(status_path, failed_status)
        failure = {
            "failed_at": failed_status["failed_at"],
            "opening_status": "OPENED_ONCE_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_forbidden": True,
            "verdict": "STOP_RANDOM_INTEGRITY",
            "test_opened": False,
        }
        failure_path = target / "failure.json"
        _json_dump_durable(failure_path, failure)
        failure_outputs = {
            str(path.relative_to(target)): file_sha256(path)
            for path in sorted(target.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        if CANONICAL_LEDGER_PATH.exists():
            failure_outputs["OPENING_LEDGER.json"] = file_sha256(
                CANONICAL_LEDGER_PATH
            )
        failure_manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: str(path) for name, path in paths.items()},
            "outputs": failure_outputs,
            "verdict": "STOP_RANDOM_INTEGRITY",
            "failure": failure,
            "invariants": {
                "opening_count": 1,
                "retry_forbidden": True,
                "winner_modified": False,
                "threshold_modified": False,
                "test_opened": False,
            },
        }
        _json_dump_durable(target / "manifest.json", failure_manifest)
        raise
    return target


def parse_args() -> argparse.Namespace:
    root = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
    partition = root / "datasets/v4_8_acceptor_partitions/1c78764d5263afca"
    development = root / "experiments/v4_8_acceptor_development/f2ea5be7c1a40647"
    historical = root / "models/v4_1/f938abf6b8a87155/acceptor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/v4_8_current_acceptor_feasibility_contract.md"),
    )
    parser.add_argument(
        "--partition-manifest", type=Path, default=partition / "manifest.json"
    )
    parser.add_argument(
        "--partition-assignments",
        type=Path,
        default=partition / "partition_assignments.parquet",
    )
    parser.add_argument(
        "--current-labels",
        type=Path,
        default=root
        / "audits/v4_7_current_adjudications/4cc5420fb5da0683/current_labels.parquet",
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=development / "manifest.json",
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        default=development / "development_report.json",
    )
    parser.add_argument(
        "--winner-model",
        type=Path,
        default=development / "winner/acceptor_model.joblib",
    )
    parser.add_argument(
        "--winner-metadata",
        type=Path,
        default=development / "winner/metadata.json",
    )
    parser.add_argument(
        "--winner-freeze",
        type=Path,
        default=development / "winner_freeze.json",
    )
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=historical / "acceptor_model.joblib",
    )
    parser.add_argument(
        "--frozen-metadata",
        type=Path,
        default=historical / "metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = open_random(
        contract_path=args.contract,
        partition_manifest_path=args.partition_manifest,
        partition_assignments_path=args.partition_assignments,
        current_labels_path=args.current_labels,
        development_manifest_path=args.development_manifest,
        development_report_path=args.development_report,
        winner_model_path=args.winner_model,
        winner_metadata_path=args.winner_metadata,
        winner_freeze_path=args.winner_freeze,
        frozen_model_path=args.frozen_model,
        frozen_metadata_path=args.frozen_metadata,
        output_root=CANONICAL_OUTPUT_ROOT,
    )
    print(target)


if __name__ == "__main__":
    main()
