from __future__ import annotations

import json
import hashlib
import inspect
from pathlib import Path

import pandas as pd
import pytest

from scripts import evaluate_v412_guard as subject


def _lock(tmp_path: Path) -> Path:
    payload = {
        "schema_version": subject.LOCK_SCHEMA_VERSION,
        "purpose": subject.PURPOSE,
        "audit_verdict": subject.AUDIT_VERDICT,
        "git_commit": "fixture",
        "source_hashes": subject._source_hashes(),
        "input_paths": {
            **{
                name: str(path.resolve())
                for name, path in subject.INPUT_PATHS.items()
            },
            "allowlist": str(subject.DEFAULT_ALLOWLIST.resolve()),
            "denylist": str(subject.DEFAULT_DENYLIST.resolve()),
        },
        "input_hashes": {
            **subject.EXPECTED_INPUT_HASHES,
            "allowlist": subject.file_sha256(subject.DEFAULT_ALLOWLIST),
            "denylist": subject.file_sha256(subject.DEFAULT_DENYLIST),
        },
        "runtime": subject._runtime(),
        "threshold": subject.FIXED_THRESHOLD,
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload))
    return path


def test_lock_requires_exact_external_audit_authorization(tmp_path):
    path = _lock(tmp_path)
    lock, digest = subject.validate_execution_lock(path, verify_git=False)
    assert lock["audit_verdict"] == "GO_EVALUATE_V412_GUARD"
    assert len(digest) == 64

    payload = json.loads(path.read_text())
    payload["unexpected"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_execution_lock(path, verify_git=False)


def test_lock_rejects_pending_audit(tmp_path):
    path = _lock(tmp_path)
    payload = json.loads(path.read_text())
    payload["audit_verdict"] = "PENDING"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independent GO missing"):
        subject.validate_execution_lock(path, verify_git=False)


def test_allowlist_projection_excludes_ground_truth():
    roots, hashes = subject.validate_allowlist_and_denylist()
    assert len(roots) == 3
    assert hashes
    assert "is_ground_truth" not in subject.RANKER_COLUMNS
    assert subject.RANKER_COLUMNS == [
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "retrieval_rank",
        "ranker_score",
        "prediction_origin",
        "oof_fold",
        "ranker_rank",
    ]


def test_allowlist_and_denylist_reject_internal_mutations(
    tmp_path, monkeypatch
):
    allow = json.loads(subject.DEFAULT_ALLOWLIST.read_text())
    allow["artifacts"][1]["files"][
        "predictions_ranker_c_oof_dev.parquet"
    ] = "0" * 64
    allow_path = tmp_path / "allow.json"
    allow_path.write_text(json.dumps(allow))
    monkeypatch.setattr(subject, "DEFAULT_ALLOWLIST", allow_path)
    with pytest.raises(ValueError, match="allowlist changed"):
        subject.validate_allowlist_and_denylist()

    monkeypatch.setattr(
        subject,
        "DEFAULT_ALLOWLIST",
        Path("config/v4_12_development_inputs.json"),
    )
    deny = json.loads(subject.DEFAULT_DENYLIST.read_text())
    deny["artifacts"][0]["files"]["sealed_mapping.parquet"] = "0" * 64
    deny_path = tmp_path / "deny.json"
    deny_path.write_text(json.dumps(deny))
    monkeypatch.setattr(subject, "DEFAULT_DENYLIST", deny_path)
    with pytest.raises(ValueError, match="denylist artifact changed"):
        subject.validate_allowlist_and_denylist()


def test_expected_inputs_are_disjoint_from_consumed_challenge():
    roots, hashes = subject.validate_allowlist_and_denylist()
    for path in subject.INPUT_PATHS.values():
        resolved = path.resolve()
        assert not any(
            resolved == root or resolved.is_relative_to(root)
            for root in roots
        )
        assert subject.EXPECTED_INPUT_HASHES[
            next(name for name, value in subject.INPUT_PATHS.items() if value == path)
        ] not in hashes


def test_evaluation_cannot_run_without_lock(tmp_path):
    with pytest.raises(FileNotFoundError):
        subject.evaluate(
            execution_lock_path=tmp_path / "missing-lock.json",
            output_root=tmp_path,
        )


def test_publication_entrypoint_exposes_no_bypass():
    assert list(inspect.signature(subject.evaluate).parameters) == [
        "execution_lock_path",
        "output_root",
    ]


def test_rss_is_measured_after_serialization_before_integrity():
    source = inspect.getsource(subject.evaluate)
    assert (
        source.index("decisions.to_parquet")
        < source.index("peak_rss_bytes = _peak_rss_bytes()")
        < source.index('staging / "integrity.json"')
    )


def test_input_inventory_detects_toctou_and_checks_additional_files(
    tmp_path, monkeypatch
):
    frozen = tmp_path / "frozen.bin"
    frozen.write_bytes(b"before")
    extra = tmp_path / "lock.json"
    extra.write_bytes(b"lock")
    digest = subject.file_sha256(frozen)
    monkeypatch.setattr(subject, "INPUT_PATHS", {"frozen": frozen})
    monkeypatch.setattr(subject, "EXPECTED_INPUT_HASHES", {"frozen": digest})
    first = subject.validate_inputs(
        forbidden_roots=set(),
        forbidden_hashes=set(),
        additional_paths={"lock": extra},
    )
    extra.write_bytes(b"changed")
    second = subject.validate_inputs(
        forbidden_roots=set(),
        forbidden_hashes=set(),
        additional_paths={"lock": extra},
    )
    assert first != second
    with pytest.raises(ValueError, match="TOCTOU.*before publication"):
        subject.assert_inventory_unchanged(
            first, second, phase="before publication"
        )


def _empty_artifact_tree(root: Path) -> None:
    root.mkdir()
    for filename in {"manifest.json", *subject.OUTPUTS}:
        (root / filename).write_bytes(b"{}")


def test_artifact_validator_rejects_extra_file_before_parsing(tmp_path):
    root = tmp_path / "artifact"
    _empty_artifact_tree(root)
    (root / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="top-level files changed"):
        subject.validate_artifact(root)


def test_artifact_validator_rejects_symlink_output(tmp_path):
    root = tmp_path / "artifact"
    _empty_artifact_tree(root)
    target = tmp_path / "elsewhere"
    target.write_bytes(b"{}")
    (root / "metrics.json").unlink()
    (root / "metrics.json").symlink_to(target)
    with pytest.raises(ValueError, match="non-regular output"):
        subject.validate_artifact(root)


def test_artifact_validator_rejects_corrupted_declared_output(tmp_path):
    identity = {
        "schema_version": subject.SCHEMA_VERSION,
        "execution_lock_sha256": "a" * 64,
        "evidence_seal": {
            "manifest_sha256": subject.EXPECTED_EVIDENCE_MANIFEST_SHA256,
            "query_sha256": subject.EXPECTED_EVIDENCE_QUERY_SHA256,
            "candidate_sha256": subject.EXPECTED_EVIDENCE_CANDIDATE_SHA256,
        },
        "input_hashes": {
            "evidence_manifest": subject.EXPECTED_EVIDENCE_MANIFEST_SHA256
        },
        "source_hashes": {},
        "threshold": subject.FIXED_THRESHOLD,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = tmp_path / build_id
    root.mkdir()
    for filename in subject.OUTPUTS:
        (root / filename).write_bytes(b"corrupted")
    outputs = {
        filename: {
            "sha256": "0" * 64,
            "size_bytes": len(b"corrupted"),
            **(
                {"row_count": 7003, "columns": subject.DECISION_COLUMNS}
                if filename == "decisions.parquet"
                else {}
            ),
        }
        for filename in subject.OUTPUTS
    }
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": "fixture",
        "outputs": outputs,
        "verdict": "STOP_V412_GUARD",
        "gate": {},
        "phase_ledger": [],
        "latency_gate_evaluated": False,
        "production_certified": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash changed"):
        subject.validate_artifact(root)


def _coherent_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "population": "comparison_dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "predicted_siret": "11111111100011",
                "acceptor_target": 1,
                "acceptor_score": 0.9,
                "decision_v411": "AUTO_MATCH",
                "review_reason_v411": None,
                "direct_candidate_count": 1,
                "direct_siren_count": 1,
                "sole_direct_siret": "11111111100011",
                "sole_direct_siren": "111111111",
                "sole_direct_in_top100": True,
                "decision_v412": "AUTO_MATCH",
                "review_reason_v412": None,
                "correct_exact_siret": True,
                "input_siret_state": "ACTIVE",
                "source_segment": "train",
                "top1_siren_candidate_count": 1.0,
                "role_crm_count": 0.0,
            }
        ],
        columns=subject.DECISION_COLUMNS,
    )


def test_decision_validator_recomputes_correctness_and_both_decisions():
    frame = _coherent_decision()
    subject.validate_decision_coherence(frame)

    forged_correctness = frame.copy()
    forged_correctness.loc[0, "correct_exact_siret"] = False
    with pytest.raises(ValueError, match="correctness was not recomputed"):
        subject.validate_decision_coherence(forged_correctness)

    forged_v411 = frame.copy()
    forged_v411.loc[0, "acceptor_score"] = 0.1
    with pytest.raises(ValueError, match="V4.11 decision changed"):
        subject.validate_decision_coherence(forged_v411)

    forged_v412 = frame.copy()
    forged_v412.loc[0, "direct_candidate_count"] = 0
    forged_v412.loc[0, "sole_direct_siret"] = None
    with pytest.raises(ValueError, match="V4.12 decision changed"):
        subject.validate_decision_coherence(forged_v412)


def _write_semantically_valid_artifact(tmp_path: Path) -> Path:
    rows = []
    populations = [
        ("fit", subject.CANONICAL_COUNTS["fit"]),
        ("threshold_dev", subject.CANONICAL_COUNTS["threshold_dev"]),
        ("comparison_dev", subject.CANONICAL_COUNTS["comparison_dev"]),
    ]
    query_index = 0
    for population, count in populations:
        for local_index in range(count):
            comparison = population == "comparison_dev"
            v411_auto = comparison and local_index < 614
            v412_auto = comparison and local_index < 600
            guard_veto = comparison and 600 <= local_index < 614
            rows.append(
                {
                    "query_id": f"q{query_index}",
                    "population": population,
                    "label_kind": "MATCH_EXACT",
                    "ground_truth_siret": "11111111100011",
                    "predicted_siret": "11111111100011",
                    "acceptor_target": 1,
                    "acceptor_score": 0.9 if v411_auto else 0.1,
                    "decision_v411": (
                        "AUTO_MATCH" if v411_auto else "REVIEW"
                    ),
                    "review_reason_v411": (
                        None if v411_auto else "LOW_CONFIDENCE"
                    ),
                    "direct_candidate_count": 0 if guard_veto else 1,
                    "direct_siren_count": 0 if guard_veto else 1,
                    "sole_direct_siret": (
                        None if guard_veto else "11111111100011"
                    ),
                    "sole_direct_siren": (
                        None if guard_veto else "111111111"
                    ),
                    "sole_direct_in_top100": (
                        None if guard_veto else True
                    ),
                    "decision_v412": (
                        "AUTO_MATCH" if v412_auto else "REVIEW"
                    ),
                    "review_reason_v412": (
                        None
                        if v412_auto
                        else (
                            "NO_DIRECT_EVIDENCE"
                            if guard_veto
                            else "LOW_CONFIDENCE"
                        )
                    ),
                    "correct_exact_siret": True,
                    "input_siret_state": "ACTIVE",
                    "source_segment": "train",
                    "top1_siren_candidate_count": 1.0,
                    "role_crm_count": 0.0,
                }
            )
            query_index += 1
    decisions = pd.DataFrame(rows, columns=subject.DECISION_COLUMNS)
    metrics, segments = subject.evaluate_comparison_gate(
        decisions, enforce_canonical=True
    )
    identity = {
        "schema_version": subject.SCHEMA_VERSION,
        "execution_lock_sha256": "a" * 64,
        "evidence_seal": {
            "manifest_sha256": subject.EXPECTED_EVIDENCE_MANIFEST_SHA256,
            "query_sha256": subject.EXPECTED_EVIDENCE_QUERY_SHA256,
            "candidate_sha256": subject.EXPECTED_EVIDENCE_CANDIDATE_SHA256,
        },
        "input_hashes": {
            "evidence_manifest": subject.EXPECTED_EVIDENCE_MANIFEST_SHA256
        },
        "source_hashes": {},
        "threshold": subject.FIXED_THRESHOLD,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = tmp_path / build_id
    root.mkdir()
    decisions.to_parquet(root / "decisions.parquet", index=False)
    (root / "metrics.json").write_text(json.dumps(metrics, sort_keys=True))
    (root / "segments.json").write_text(json.dumps(segments, sort_keys=True))
    integrity = {
        "evidence_closed_before_historical_inputs": True,
        "evidence_build_id": subject.EVIDENCE_BUILD_ID,
        "split_projection": subject.SPLIT_COLUMNS,
        "ranker_projection": subject.RANKER_COLUMNS,
        "ranker_is_ground_truth_opened": False,
        "retrieval_labels_opened": False,
        "challenge_opened": False,
        "ranker_pool_cap": 100,
        "ranker_pool_modified": False,
        "model_retrained": False,
        "threshold_changed": False,
        "v412_is_pure_veto": True,
        "comparison_dev_only_gate": True,
        "peak_rss_bytes": 1,
        "peak_rss_limit_bytes": 8 * 1024**3,
    }
    (root / "integrity.json").write_text(json.dumps(integrity, sort_keys=True))
    outputs = {}
    for filename in subject.OUTPUTS:
        path = root / filename
        outputs[filename] = {
            "sha256": subject.file_sha256(path),
            "size_bytes": path.stat().st_size,
            **(
                {
                    "row_count": len(decisions),
                    "columns": subject.DECISION_COLUMNS,
                }
                if filename == "decisions.parquet"
                else {}
            ),
        }
    gate = {
        "comparison_dev_only": True,
        "minimum_auto": 600,
        "maximum_error_auto": 0,
        "maximum_ambiguous_auto": 0,
        "segment_minimum_rows": 100,
        "segment_maximum_coverage_loss_points": 2,
    }
    phases = [
        {
            "phase": "EVIDENCE_SEAL_VALIDATED",
            "hashes": identity["evidence_seal"],
        },
        {
            "phase": "POST_EVIDENCE_SEAL",
            "deserialized": [
                "split_assignments.parquet:SPLIT_COLUMNS",
                "predictions_ranker_c_oof_dev.parquet:RANKER_COLUMNS",
                "acceptor_scenes.parquet:SCENE_METADATA+FEATURE_ORDER",
                "acceptor_model.joblib",
            ],
        },
    ]
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": "fixture",
        "outputs": outputs,
        "verdict": metrics["verdict"],
        "gate": gate,
        "phase_ledger": phases,
        "latency_gate_evaluated": False,
        "production_certified": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


def test_artifact_semantic_mutation_fails_even_after_internal_rehash(tmp_path):
    root = _write_semantically_valid_artifact(tmp_path)
    subject.validate_artifact(root)
    decisions_path = root / "decisions.parquet"
    decisions = pd.read_parquet(decisions_path)
    decisions.loc[0, "correct_exact_siret"] = False
    decisions.to_parquet(decisions_path, index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["decisions.parquet"]["sha256"] = subject.file_sha256(
        decisions_path
    )
    manifest["outputs"]["decisions.parquet"][
        "size_bytes"
    ] = decisions_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="correctness was not recomputed"):
        subject.validate_artifact(root)


def test_fsync_helpers_cover_files_and_directories(tmp_path):
    path = tmp_path / "durable"
    path.write_bytes(b"payload")
    subject._fsync_file(path)
    subject._fsync_directory(tmp_path)
