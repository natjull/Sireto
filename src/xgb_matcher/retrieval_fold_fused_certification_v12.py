"""Post-fit certification for one immutable fold-fused v12 OOF run.

The OOF executor is intentionally unable to consume POSTFIT parity: those
models do not exist until the run completes.  This joiner verifies the exact
OOF manifest and its three model bytes against the later parity replay, then
publishes a no-refit certificate.  Runtime remains a subsequent, separately
sealed gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import os
import tempfile

from .retrieval_fold_fused_oof_v12 import (
    FoldFusedInvariantError,
    MANIFEST_SCHEMA_VERSION,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_runtime_v12 import load_parity_evidence_v12


CERTIFICATION_SCHEMA_VERSION = (
    "sireto-retrieval-v12-fold-fused-postfit-certificate-1"
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _resolve_hashed_artifact(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise FoldFusedInvariantError(f"OOF artifact hash mismatch: {label}")
    return path


def _validate_oof_manifest(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": "NESTED_OUTER_OOF_FOLD_FUSED",
        "status": "EXECUTED_NO_READY_CLAIM",
        "opened_folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "logical_recipe_changes": False,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise FoldFusedInvariantError("OOF manifest is not certifiable")
    parity = raw.get("parity") or {}
    gates = raw.get("gates") or {}
    lineage = raw.get("lineage") or {}
    if (
        parity.get("status") != "NOT_RUN"
        or gates.get("parity_gate") != "NOT_RUN"
        or gates.get("decision") != "NO_READY_DECISION"
        or gates.get("gate_eligible") is not False
        or lineage.get("parity_evidence_sha256") is not None
    ):
        raise FoldFusedInvariantError(
            "OOF must remain PREFIT-only/NO_READY before postfit certification"
        )
    models = raw.get("site_models")
    if not isinstance(models, Mapping) or set(models) != {
        str(value) for value in TRAIN_FOLDS
    }:
        raise FoldFusedInvariantError("OOF manifest does not seal three site models")
    for outer in TRAIN_FOLDS:
        record = models[str(outer)]
        if not isinstance(record, Mapping):
            raise FoldFusedInvariantError("invalid OOF site model record")
        _resolve_hashed_artifact(path, record, label=f"site model {outer}")
    for section in ("train_matrices", "outputs"):
        records = raw.get(section)
        if not isinstance(records, Mapping) or not records:
            raise FoldFusedInvariantError(f"OOF manifest has no {section}")
        for key, record in records.items():
            if not isinstance(record, Mapping):
                raise FoldFusedInvariantError(f"invalid OOF {section} record")
            _resolve_hashed_artifact(path, record, label=f"{section}/{key}")
    return raw


def certify_fold_fused_oof_postfit_v12(
    *,
    oof_manifest_path: Path,
    postfit_parity_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Bind exact POSTFIT parity to one OOF run, without fitting anything."""

    oof_manifest_path = Path(oof_manifest_path).resolve()
    postfit_parity_manifest_path = Path(postfit_parity_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    oof = _validate_oof_manifest(oof_manifest_path)
    oof_sha = sha256_file(oof_manifest_path)
    lineage = oof["lineage"]
    parity_lineage_keys = (
        "scene_ir_sha256",
        "scene_ir_manifest_sha256",
        "source_candidate_sha256",
        "source_manifest_sha256",
        "label_ledger_sha256",
        "label_ledger_manifest_sha256",
        "scientific_config_sha256",
        "executor_config_sha256",
        "official_sidecar_manifest_sha256",
        "mmap_sidecar_manifest_sha256",
        "reader_parity_evidence_sha256",
        "parent_model_bundle_sha256",
        "query_sample_sha256",
    )
    expected_lineage = {
        key: str(lineage.get(key) or "") for key in parity_lineage_keys
    }
    if any(len(value) != 64 for value in expected_lineage.values()):
        raise FoldFusedInvariantError("OOF lineage required by parity is incomplete")
    _evidence, parity_sha = load_parity_evidence_v12(
        postfit_parity_manifest_path,
        expected_lineage=expected_lineage,
    )
    parity = json.loads(postfit_parity_manifest_path.read_text(encoding="utf-8"))
    if parity.get("site_model_manifest_sha256") != oof_sha:
        raise FoldFusedInvariantError(
            "POSTFIT parity was not run against this exact OOF manifest"
        )
    oof_models = oof["site_models"]
    parity_models = parity.get("site_model_sha256") or {}
    parity_recipes = parity.get("site_model_recipe_sha256") or {}
    for outer in TRAIN_FOLDS:
        record = oof_models[str(outer)]
        if (
            parity_models.get(str(outer)) != record.get("sha256")
            or parity_recipes.get(str(outer)) != record.get("recipe_sha256")
        ):
            raise FoldFusedInvariantError(
                f"POSTFIT parity/site model binding mismatch: outer {outer}"
            )

    gates = oof["gates"]
    if gates.get("prerank_recall_at_500", {}).get("rate", 0.0) < gates.get(
        "thresholds", {}
    ).get("prerank_recall_at_500", float("inf")):
        decision = "PIVOT_GENERATION"
    elif gates.get("recall_gates_pass") is not True:
        decision = "PIVOT_ADMISSION"
    else:
        decision = "PENDING_RUNTIME_PROFILE"
    certificate = {
        "schema_version": CERTIFICATION_SCHEMA_VERSION,
        "status": "POSTFIT_PARITY_CERTIFIED_RUNTIME_PENDING",
        "decision": decision,
        "gate_eligible": False,
        "refit_performed": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "runtime_gate": "NOT_EVALUATED",
        "master_decision_blocking": False,
        "lineage": {
            "oof_manifest_sha256": oof_sha,
            "postfit_parity_manifest_sha256": parity_sha,
            "prefit_parity_evidence_sha256": lineage.get(
                "prefit_parity_evidence_sha256"
            ),
            "reader_parity_evidence_sha256": lineage.get(
                "reader_parity_evidence_sha256"
            ),
            "site_model_sha256": {
                str(outer): oof_models[str(outer)]["sha256"]
                for outer in TRAIN_FOLDS
            },
            "site_model_recipe_sha256": {
                str(outer): oof_models[str(outer)]["recipe_sha256"]
                for outer in TRAIN_FOLDS
            },
        },
        "gates": {
            "master_recall_at_2000": gates.get("master_recall_at_2000"),
            "master_headroom_target_met": gates.get("master_headroom_target_met"),
            "prerank_recall_at_500": gates.get("prerank_recall_at_500"),
            "exact_recall_at_100": gates.get("exact_recall_at_100"),
            "exact_recall_at_100_by_outer_fold": gates.get(
                "exact_recall_at_100_by_outer_fold"
            ),
            "recall_gates_pass": gates.get("recall_gates_pass"),
            "postfit_parity_gate": "PASS",
            "latency_gate": "NOT_EVALUATED",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(_canonical_json(certificate) + "\n", encoding="utf-8")
        os.rename(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


__all__ = [
    "CERTIFICATION_SCHEMA_VERSION",
    "certify_fold_fused_oof_postfit_v12",
]
