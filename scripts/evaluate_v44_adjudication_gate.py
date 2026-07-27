#!/usr/bin/env python3
"""Evaluate the frozen V4.4 autonomous-adjudication retraining gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v44_adjudications import (  # noqa: E402
    SCHEMA_VERSION as ADJUDICATION_SCHEMA_VERSION,
    validate_artifact as validate_adjudication_artifact,
    validate_canonical_adjudications,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-adjudication-gate-1"
POLICY_VERSION = "v4.4-autonomous-adjudication-gate-v1"
VERDICTS = {
    "GO_RETRAIN_AUTO",
    "PIVOT_MORE_EVIDENCE",
    "STOP_AUTONOMOUS_LABELING",
}
THRESHOLDS = {
    "top1_correct_evidence_validated_min": 75,
    "top1_wrong_evidence_validated_min": 50,
    "random_evidence_validated_min": 30,
    "forbidden_proof_decision_max": 0,
}
RANDOM_STRATUM = "RANDOM_POPULATION"
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "v4_4_autonomous_adjudication_contract.md"
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _row_fingerprint(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    payload = {
        column: _canonical_value(row.get(column))
        for column in sorted(columns)
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def load_adjudication_artifacts(
    artifact_dirs: Iterable[Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate, merge and safely deduplicate canonical adjudication artifacts."""

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw in artifact_dirs:
        path = Path(raw).resolve()
        if path not in seen_paths:
            unique_paths.append(path)
            seen_paths.add(path)
    if not unique_paths:
        raise ValueError("At least one adjudication artifact is required")

    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    canonical_columns: set[str] | None = None
    for path in unique_paths:
        validate_adjudication_artifact(path)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported adjudication schema: {path}")
        frame = pd.read_parquet(path / "adjudications.parquet")
        validate_canonical_adjudications(frame)
        columns = set(frame.columns)
        if canonical_columns is None:
            canonical_columns = columns
        elif columns != canonical_columns:
            raise ValueError("Adjudication artifacts expose different canonical schemas")
        frames.append(frame)
        provenance.append(
            {
                "path": str(path),
                "build_id": _text(manifest.get("build_id")),
                "manifest_sha256": file_sha256(manifest_path),
                "adjudications_sha256": file_sha256(
                    path / "adjudications.parquet"
                ),
                "case_count": int(len(frame)),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("The adjudication corpus is empty")
    columns = list(combined.columns)
    records: list[dict[str, Any]] = []
    for case_id, group in combined.groupby("audit_case_id", sort=True):
        fingerprints = {
            _row_fingerprint(row, columns)
            for row in group.to_dict("records")
        }
        if len(fingerprints) != 1:
            raise ValueError(
                f"Conflicting adjudications for audit_case_id={case_id}"
            )
        records.append(group.iloc[0].to_dict())
    deduplicated = pd.DataFrame(records, columns=columns).sort_values(
        "audit_case_id", kind="stable"
    ).reset_index(drop=True)
    validate_canonical_adjudications(deduplicated)
    return deduplicated, sorted(
        provenance,
        key=lambda item: (item["manifest_sha256"], item["path"]),
    )


def compute_gate_metrics(adjudications: pd.DataFrame) -> dict[str, Any]:
    """Compute contract counts from an already validated canonical corpus."""

    validate_canonical_adjudications(adjudications)
    validated = adjudications["evidence_validated"].astype(bool)
    labels = adjudications["adjudication_label"].astype(str)
    random = adjudications["sampling_stratum"].fillna("").astype(str).eq(
        RANDOM_STRATUM
    )
    label_counts = {
        label: int(labels.eq(label).sum())
        for label in (
            "TOP1_CORRECT",
            "TOP1_WRONG",
            "AMBIGUOUS",
            "UNRESOLVED",
        )
    }
    validated_label_counts = {
        label: int((labels.eq(label) & validated).sum())
        for label in (
            "TOP1_CORRECT",
            "TOP1_WRONG",
            "AMBIGUOUS",
            "UNRESOLVED",
        )
    }
    # Every input artifact is recomputed through the canonical builder. That
    # builder refuses model/address-only proof kinds and collapses all
    # SIRENE-derived views into one independence group.
    forbidden_proof_decision_count = 0
    return {
        "unique_case_count": int(len(adjudications)),
        "label_counts": label_counts,
        "evidence_validated_count": int(validated.sum()),
        "validated_label_counts": validated_label_counts,
        "top1_correct_evidence_validated_count": validated_label_counts[
            "TOP1_CORRECT"
        ],
        "top1_wrong_evidence_validated_count": validated_label_counts[
            "TOP1_WRONG"
        ],
        "random_case_count": int(random.sum()),
        "random_evidence_validated_count": int((random & validated).sum()),
        "unresolved_count": label_counts["UNRESOLVED"],
        "acceptor_eligible_count": int(
            adjudications["acceptor_eligible"].astype(bool).sum()
        ),
        "ranker_eligible_count": int(
            adjudications["ranker_eligible"].astype(bool).sum()
        ),
        "forbidden_proof_decision_count": forbidden_proof_decision_count,
    }


def decide_gate(
    metrics: Mapping[str, Any],
    *,
    stop_requested: bool = False,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered gate without inferring STOP from scarcity."""

    reason = _text(stop_reason)
    if stop_requested and not reason:
        raise ValueError("STOP_AUTONOMOUS_LABELING requires an explicit reason")
    if reason and not stop_requested:
        raise ValueError("A stop reason requires stop_requested=true")

    checks = {
        "top1_correct_evidence_validated": (
            int(metrics["top1_correct_evidence_validated_count"])
            >= THRESHOLDS["top1_correct_evidence_validated_min"]
        ),
        "top1_wrong_evidence_validated": (
            int(metrics["top1_wrong_evidence_validated_count"])
            >= THRESHOLDS["top1_wrong_evidence_validated_min"]
        ),
        "random_evidence_validated": (
            int(metrics["random_evidence_validated_count"])
            >= THRESHOLDS["random_evidence_validated_min"]
        ),
        "zero_forbidden_proof_decisions": (
            int(metrics["forbidden_proof_decision_count"])
            <= THRESHOLDS["forbidden_proof_decision_max"]
        ),
    }
    deficits = {
        "top1_correct_evidence_validated": max(
            0,
            THRESHOLDS["top1_correct_evidence_validated_min"]
            - int(metrics["top1_correct_evidence_validated_count"]),
        ),
        "top1_wrong_evidence_validated": max(
            0,
            THRESHOLDS["top1_wrong_evidence_validated_min"]
            - int(metrics["top1_wrong_evidence_validated_count"]),
        ),
        "random_evidence_validated": max(
            0,
            THRESHOLDS["random_evidence_validated_min"]
            - int(metrics["random_evidence_validated_count"]),
        ),
    }
    if stop_requested:
        verdict = "STOP_AUTONOMOUS_LABELING"
        rationale = reason
        source_status = "EXPLICITLY_EXHAUSTED"
    elif all(checks.values()):
        verdict = "GO_RETRAIN_AUTO"
        rationale = "All pre-registered evidence gates are satisfied."
        source_status = "OPEN_OR_SUFFICIENT"
    else:
        verdict = "PIVOT_MORE_EVIDENCE"
        rationale = (
            "The corpus is currently insufficient; public evidence collection "
            "remains open."
        )
        source_status = "OPEN"
    if verdict not in VERDICTS:
        raise AssertionError("Unexpected V4.4 gate verdict")
    return {
        "verdict": verdict,
        "rationale": rationale,
        "source_status": source_status,
        "stop_requested": bool(stop_requested),
        "stop_reason": reason or None,
        "checks": checks,
        "deficits": deficits,
        "thresholds": dict(THRESHOLDS),
    }


def evaluate_adjudication_gate(
    adjudications: pd.DataFrame,
    *,
    stop_requested: bool = False,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    metrics = compute_gate_metrics(adjudications)
    decision = decide_gate(
        metrics,
        stop_requested=stop_requested,
        stop_reason=stop_reason,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        **decision,
        "metrics": metrics,
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    checks = report["checks"]
    deficits = report["deficits"]
    rationale = _text(report["rationale"]).replace("\n", " ").replace("|", "\\|")
    rows = [
        (
            "TOP1_CORRECT validés",
            metrics["top1_correct_evidence_validated_count"],
            report["thresholds"]["top1_correct_evidence_validated_min"],
            checks["top1_correct_evidence_validated"],
            deficits["top1_correct_evidence_validated"],
        ),
        (
            "TOP1_WRONG validés",
            metrics["top1_wrong_evidence_validated_count"],
            report["thresholds"]["top1_wrong_evidence_validated_min"],
            checks["top1_wrong_evidence_validated"],
            deficits["top1_wrong_evidence_validated"],
        ),
        (
            "Cas random validés",
            metrics["random_evidence_validated_count"],
            report["thresholds"]["random_evidence_validated_min"],
            checks["random_evidence_validated"],
            deficits["random_evidence_validated"],
        ),
        (
            "Décisions sur preuve interdite",
            metrics["forbidden_proof_decision_count"],
            report["thresholds"]["forbidden_proof_decision_max"],
            checks["zero_forbidden_proof_decisions"],
            0,
        ),
    ]
    table = "\n".join(
        f"| {name} | {value} | {threshold} | "
        f"{'PASS' if passed else 'FAIL'} | {deficit} |"
        for name, value, threshold, passed, deficit in rows
    )
    return (
        "# Gate V4.4 — adjudications autonomes\n\n"
        f"Verdict : **`{report['verdict']}`**\n\n"
        f"{rationale}\n\n"
        "## Gate\n\n"
        "| Contrôle | Observé | Seuil | Statut | Manque |\n"
        "|---|---:|---:|---|---:|\n"
        f"{table}\n\n"
        "## Corpus dédupliqué\n\n"
        f"- Cas uniques : {metrics['unique_case_count']}\n"
        f"- Preuves validées : {metrics['evidence_validated_count']}\n"
        f"- `UNRESOLVED` : {metrics['unresolved_count']}\n"
        f"- Éligibles accepteur : {metrics['acceptor_eligible_count']}\n"
        f"- Éligibles ranker : {metrics['ranker_eligible_count']}\n"
        f"- Statut des sources : `{report['source_status']}`\n"
    )


def build_gate_artifact(
    *,
    adjudication_artifact_dirs: Iterable[Path],
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    stop_requested: bool = False,
    stop_reason: str | None = None,
) -> Path:
    """Validate inputs, evaluate the gate and write an immutable report."""

    adjudications, provenance = load_adjudication_artifacts(
        adjudication_artifact_dirs
    )
    contract_path = Path(contract_path).resolve()
    contract_hash = file_sha256(contract_path)
    # Validate STOP arguments before creating a target directory.
    report = evaluate_adjudication_gate(
        adjudications,
        stop_requested=stop_requested,
        stop_reason=stop_reason,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "contract_sha256": contract_hash,
        "adjudication_manifest_hashes": sorted(
            item["manifest_sha256"] for item in provenance
        ),
        "stop_requested": bool(stop_requested),
        "stop_reason": _text(stop_reason) or None,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.4 gate report exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        evaluated_at = datetime.now(timezone.utc).isoformat()
        report = {
            **report,
            "build_id": build_id,
            "evaluated_at": evaluated_at,
            "contract": {
                "path": str(contract_path),
                "sha256": contract_hash,
            },
            "input_artifacts": provenance,
            "input_artifact_count": len(provenance),
            "input_row_count": sum(item["case_count"] for item in provenance),
            "deduplicated_case_count": int(len(adjudications)),
        }
        json_path = staging / "gate_report.json"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        markdown_path = staging / "gate_report.md"
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")
        outputs = {
            json_path.name: file_sha256(json_path),
            markdown_path.name: file_sha256(markdown_path),
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": evaluated_at,
            "inputs": provenance,
            "contract": {
                "path": str(contract_path),
                "sha256": contract_hash,
            },
            "outputs": outputs,
            "verdict": report["verdict"],
            "invariants": {
                "input_artifacts_validated_and_recomputed": True,
                "duplicate_case_conflicts_blocking": True,
                "insufficient_defaults_to_pivot": True,
                "stop_requires_explicit_option_and_reason": True,
                "no_model_training_or_inference": True,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_gate_artifact(artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported V4.4 gate artifact schema")
    if manifest.get("verdict") not in VERDICTS:
        raise ValueError("Invalid V4.4 gate verdict")
    for filename, expected_hash in (manifest.get("outputs") or {}).items():
        if file_sha256(artifact_dir / filename) != expected_hash:
            raise ValueError(f"V4.4 gate output hash mismatch: {filename}")
    report = json.loads(
        (artifact_dir / "gate_report.json").read_text(encoding="utf-8")
    )
    if report.get("verdict") != manifest.get("verdict"):
        raise ValueError("Gate report and manifest verdicts differ")
    if report.get("build_id") != manifest.get("build_id"):
        raise ValueError("Gate report and manifest build IDs differ")
    contract = manifest.get("contract") or {}
    contract_path = Path(contract.get("path") or "")
    if (
        not contract_path.is_file()
        or file_sha256(contract_path) != contract.get("sha256")
    ):
        raise ValueError("V4.4 gate contract hash mismatch")
    input_paths = [
        Path(item.get("path") or "")
        for item in manifest.get("inputs") or []
    ]
    adjudications, provenance = load_adjudication_artifacts(input_paths)
    expected = evaluate_adjudication_gate(
        adjudications,
        stop_requested=bool(manifest.get("stop_requested")),
        stop_reason=manifest.get("stop_reason"),
    )
    for field in (
        "verdict",
        "rationale",
        "source_status",
        "stop_requested",
        "stop_reason",
        "checks",
        "deficits",
        "thresholds",
        "metrics",
    ):
        if report.get(field) != expected.get(field):
            raise ValueError(f"V4.4 gate report field mismatch: {field}")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "contract_sha256": contract["sha256"],
        "adjudication_manifest_hashes": sorted(
            item["manifest_sha256"] for item in provenance
        ),
        "stop_requested": bool(manifest.get("stop_requested")),
        "stop_reason": _text(manifest.get("stop_reason")) or None,
    }
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if expected_build_id != manifest.get("build_id"):
        raise ValueError("V4.4 gate build ID mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adjudication-artifact",
        type=Path,
        action="append",
        dest="artifacts",
        help="Repeat for every immutable adjudication artifact.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--stop-autonomous-labeling", action="store_true")
    parser.add_argument("--stop-reason")
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact is not None:
        validate_gate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    if not args.artifacts or args.output_root is None:
        raise SystemExit(
            "Building requires --adjudication-artifact and --output-root"
        )
    print(
        build_gate_artifact(
            adjudication_artifact_dirs=args.artifacts,
            output_root=args.output_root,
            contract_path=args.contract,
            stop_requested=args.stop_autonomous_labeling,
            stop_reason=args.stop_reason,
        )
    )


if __name__ == "__main__":
    main()
