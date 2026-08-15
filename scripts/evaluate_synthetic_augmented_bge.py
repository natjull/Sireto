#!/usr/bin/env python3
"""Compare the published real-only BGE to one synthetic-augmented fold-0 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_synthetic_augmented_xgb import (  # noqa: E402
    _metrics,
    _operational_correct,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_CONTROL = BASE / "experiments/v4_12_bge_groupwise/01e1049c16af2600"
DEFAULT_REAL_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_REAL_TEXT = BASE / "datasets/v4_12_neural_text_corpus/02b8668f8050c5e9"
DEFAULT_PLAN = Path("config/synthetic_augmented_model_eval_v1.json")
DEFAULT_OUTPUT_ROOT = BASE / "experiments/synthetic_augmented_bge_comparison_v1"
SCHEMA_VERSION = "sireto-synthetic-augmented-bge-comparison-1"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_bge(root: Path, *, require_weighted: bool) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    for name in ("target_top1_detail.parquet", "target_metrics.csv", "evaluation.json"):
        if manifest.get("outputs", {}).get(name) != file_sha256(root / name):
            raise ValueError(f"BGE artifact hash mismatch: {root / name}")
    identity = manifest.get("build_identity", {})
    if identity.get("train_folds") != [2, 3, 4] or int(identity.get("target_fold", -1)) != 0:
        raise ValueError("BGE artifact is not the folds 2/3/4 to fold-0 experiment")
    if evaluation.get("scope") != "OOF_TARGET" or evaluation.get("positive_injection") is not False:
        raise ValueError("BGE artifact is not a complete non-injected fold-0 run")
    if require_weighted and not evaluation.get("scene_weight", {}).get("weighted_groupwise_loss"):
        raise ValueError("Candidate BGE run did not apply scene-level weights")
    return manifest


def run(args: argparse.Namespace) -> Path:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    control_manifest = _validate_bge(args.control, require_weighted=False)
    candidate_manifest = _validate_bge(args.candidate, require_weighted=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(args.plan),
        "control_manifest_sha256": file_sha256(args.control / "manifest.json"),
        "candidate_manifest_sha256": file_sha256(args.candidate / "manifest.json"),
        "real_business_manifest_sha256": file_sha256(args.real_business / "manifest.json"),
        "real_text_manifest_sha256": file_sha256(args.real_text / "manifest.json"),
        "control_model_revision": control_manifest.get("build_identity", {}).get("model_revision"),
        "candidate_model_revision": candidate_manifest.get("build_identity", {}).get("model_revision"),
        "development_fold": 0,
    }
    if identity["control_model_revision"] != identity["candidate_model_revision"]:
        raise ValueError("BGE arms do not use the same pinned model revision")
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    details: list[pd.DataFrame] = []
    results: dict[str, Any] = {}
    for arm, root in (("REAL_ONLY_PUBLISHED", args.control), ("REAL_PLUS_SYNTHETIC", args.candidate)):
        detail = pd.read_parquet(root / "target_top1_detail.parquet")
        detail["query_id"] = detail["query_id"].astype(str)
        detail["candidate_siret"] = detail["candidate_siret"].astype("string").str.zfill(14)
        detail["ground_truth_siret"] = detail["ground_truth_siret"].astype("string").str.zfill(14)
        detail["exact_siret_correct"] = detail["candidate_siret"].eq(
            detail["ground_truth_siret"]
        ).fillna(False)
        detail["operational_siret_correct"] = _operational_correct(
            predictions=detail[["query_id", "candidate_siret"]],
            labels=detail,
            real_business=args.real_business,
            real_text=args.real_text,
        )
        detail["arm"] = arm
        results[arm] = {
            "exact_metrics": _metrics(detail, "exact_siret_correct"),
            "operational_metrics_secondary": _metrics(detail, "operational_siret_correct"),
        }
        details.append(detail)

    combined = pd.concat(details, ignore_index=True)
    pivot = combined.pivot(index="query_id", columns="arm", values="exact_siret_correct")
    matrix = {
        "both_correct": int((pivot["REAL_ONLY_PUBLISHED"] & pivot["REAL_PLUS_SYNTHETIC"]).sum()),
        "real_only_correct": int((pivot["REAL_ONLY_PUBLISHED"] & ~pivot["REAL_PLUS_SYNTHETIC"]).sum()),
        "synthetic_only_correct": int((~pivot["REAL_ONLY_PUBLISHED"] & pivot["REAL_PLUS_SYNTHETIC"]).sum()),
        "both_wrong": int((~pivot["REAL_ONLY_PUBLISHED"] & ~pivot["REAL_PLUS_SYNTHETIC"]).sum()),
    }
    base = {row["segment"]: row for row in results["REAL_ONLY_PUBLISHED"]["exact_metrics"]}
    candidate = {row["segment"]: row for row in results["REAL_PLUS_SYNTHETIC"]["exact_metrics"]}
    frozen = plan["development_gate"]
    gate = {
        "exact_gain_at_least_10": candidate["exact"]["correct"] - base["exact"]["correct"]
        >= int(frozen["minimum_exact_gain_over_paired_control"]),
        "exact_at_least_2452": candidate["exact"]["correct"] >= int(frozen["minimum_exact_correct"]),
        "difficult_at_least_33": candidate["difficult"]["correct"] >= int(frozen["minimum_difficult_correct"]),
        "active_at_least_2164": candidate["active"]["correct"] >= int(frozen["minimum_active_correct"]),
        "closed_at_least_246": candidate["closed"]["correct"] >= int(frozen["minimum_closed_correct"]),
        "evaluation_real_fold0_only": True,
        "confirmation_fold_closed": True,
        "final_test_closed": True,
        "positive_injection": False,
    }
    passed = all(gate.values())

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        combined.to_parquet(temporary / "fold0_top1_detail.parquet", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "DEVELOPMENT_FOLD_0_PAIRED",
            "arms": results,
            "paired_exact_matrix": matrix,
            "gate": gate,
            "gate_passed": passed,
            "verdict": "GO_SYNTHETIC_AUGMENTATION_BGE" if passed else "STOP_SYNTHETIC_AUGMENTATION_BGE",
            "primary_metric": "exact_siret_hit_at_1",
            "operational_metric_is_secondary": True,
            "risk_model_trained": False,
            "calibration_trained": False,
            "auto_threshold_selected": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        outputs = {
            str(path.relative_to(temporary)): file_sha256(path)
            for path in temporary.rglob("*") if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "positive_injection": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
            "outputs": outputs,
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--real-business", type=Path, default=DEFAULT_REAL_BUSINESS)
    parser.add_argument("--real-text", type=Path, default=DEFAULT_REAL_TEXT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
