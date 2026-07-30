from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load(
    "independent_audit_v412_unit_retrieval_evaluation",
    ROOT / "scripts/independent_audit_v412_unit_retrieval_evaluation.py",
)
evaluator = _load(
    "evaluate_v412_unit_retrieval_for_independent_test",
    ROOT / "scripts/evaluate_v412_unit_retrieval.py",
)
fixtures = _load(
    "evaluator_test_fixtures",
    ROOT / "tests/test_evaluate_v412_unit_retrieval.py",
)


def _append_final_attempt(
    plan: dict,
    attempt_dir: Path,
    evaluator_build_id: str,
    computed_attestation_sha256: str,
) -> None:
    for index, (state, phase) in enumerate(
        (
            ("STARTED", "ORACLE_OPEN_COMMITTED"),
            ("RECOVERABLE", "COMPUTED_STAGING_VALID"),
            ("RECOVERABLE", "PENDING_BOTH_VALID"),
            ("RECOVERABLE", "AUDIT_FINAL"),
            ("RECOVERABLE", "EVALUATION_FINAL"),
            ("FINAL", "TERMINAL"),
        ),
        start=1,
    ):
        evaluator.append_event(
            attempt_dir,
            plan["attempt_protocol"],
            state=state,
            phase=phase,
            oracle_open_committed=True,
            evaluator_build_id=evaluator_build_id,
            computed_attestation_sha256=(
                computed_attestation_sha256
                if phase == "COMPUTED_STAGING_VALID"
                else ...
            ),
            reason_code=None,
            now=lambda index=index: f"t{index}",
        )


def _authorized_artifacts(tmp_path: Path):
    plan, worker_spec, fds = fixtures._worker_fixture(tmp_path)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    for role in set(plan["input_paths"]) - set(worker_spec["input_paths"]):
        path = runtime_root / role
        path.write_bytes(f"synthetic-{role}".encode())
        plan["input_paths"][role] = str(path)
        plan["input_hashes"][role] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan["input_paths"].update(worker_spec["input_paths"])
    plan["input_hashes"].update(worker_spec["input_hashes"])
    plan["future_sources"] = []
    plan_path = tmp_path / "synthetic-plan.json"
    plan_path.write_bytes(evaluator.canonical_json(plan))
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    sandbox = {
        "executable": plan["input_paths"]["sandbox_executable"],
        "executable_sha256": plan["input_hashes"]["sandbox_executable"],
        "python_framework_app": plan["input_paths"]["python_framework_app"],
        "python_framework_app_sha256": plan["input_hashes"][
            "python_framework_app"
        ],
        "python_framework_library": plan["input_paths"][
            "python_framework_library"
        ],
        "python_framework_library_sha256": plan["input_hashes"][
            "python_framework_library"
        ],
        "git_executable": plan["input_paths"]["git_executable"],
        "git_executable_sha256": plan["input_hashes"]["git_executable"],
        "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
        "device_read_paths": ["/dev/null", "/dev/urandom", "/dev/fd"],
        "network_allowed": False,
        "fork_allowed": False,
        "write_scope": "PRIVATE_EVALUATOR_STAGING_ONLY",
    }
    lock = {
        "schema_version": evaluator.LOCK_SCHEMA,
        "purpose": evaluator.LOCK_PURPOSE,
        "audit_verdict": evaluator.LOCK_VERDICT,
        "git_commit": commit,
        "source_hashes": {},
        "input_paths": plan["input_paths"],
        "input_hashes": plan["input_hashes"],
        "evaluation_spec_sha256": hashlib.sha256(
            evaluator.canonical_json(plan["evaluation_spec"])
        ).hexdigest(),
        "runtime": plan["runtime"],
        "outputs": plan["outputs"],
        "sandbox": sandbox,
        "max_rss_bytes": plan["max_rss_bytes"],
    }
    lock_path = tmp_path / "synthetic-lock.json"
    lock_path.write_bytes(evaluator.canonical_json(lock))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    identity = {
        "schema_version": plan["identity_projections"][
            "build_identity_schema_version"
        ],
        "plan_sha256": plan_sha,
        "lock_sha256": lock_sha,
        "source_hashes": {},
        "input_hashes": plan["input_hashes"],
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
        "evaluation_spec": plan["evaluation_spec"],
        "runtime": plan["runtime"],
    }
    evaluator_build_id = hashlib.sha256(
        evaluator.canonical_json(identity)
    ).hexdigest()
    attempt_dir, _slot, attempt_id = evaluator.ensure_receipt(
        plan, lock, plan_sha, lock_sha, now=lambda: "t0"
    )
    worker_spec.update(
        {
            "evaluator_build_id": evaluator_build_id,
            "attempt_id": attempt_id,
            "input_hashes": plan["input_hashes"],
            "git_commit": commit,
            "source_hashes": {},
            "plan_sha256": plan_sha,
            "lock_sha256": lock_sha,
        }
    )
    evaluation, audit_root = evaluator.worker_execute(worker_spec)
    attestation_sha, _validation = fixtures._write_computed_attestation(
        plan,
        worker_spec,
        fds,
        attempt_dir,
        evaluation,
        audit_root,
    )
    _append_final_attempt(
        plan, attempt_dir, evaluator_build_id, attestation_sha
    )
    return (
        plan,
        worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    )


def _rewrite_attestation_and_authorization(
    plan: dict,
    attempt_dir: Path,
    attestation: dict,
) -> str:
    attestation_path = attempt_dir / "computed_attestation.json"
    os.chmod(attestation_path, 0o600)
    attestation_path.write_bytes(evaluator.canonical_json(attestation))
    os.chmod(attestation_path, 0o400)
    digest = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    events_path = attempt_dir / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    previous = None
    output = bytearray()
    seen_computed = False
    for event in events:
        if event["phase"] == "COMPUTED_STAGING_VALID":
            seen_computed = True
        if seen_computed:
            event["computed_attestation_sha256"] = digest
        event["previous_event_sha256"] = previous
        line = evaluator.canonical_json(event)
        output.extend(line)
        previous = hashlib.sha256(line).hexdigest()
    os.chmod(events_path, 0o600)
    events_path.write_bytes(bytes(output))
    state_path = attempt_dir / "state.json"
    os.chmod(state_path, 0o600)
    state_path.unlink()
    evaluator.recover_state_cache(attempt_dir, plan["attempt_protocol"])
    return digest


def test_static_audit_is_independent_and_closed() -> None:
    result = audit.static_audit(ROOT)
    assert result["verdict"] == "GO_CODE_V412_UNIT_RETRIEVAL_EVALUATOR_AUDIT"
    source = (
        ROOT / "scripts/independent_audit_v412_unit_retrieval_evaluation.py"
    ).read_text()
    assert "import evaluate_v412_unit_retrieval" not in source
    for marker in (
        "computed_attestation.json",
        "computed_attestation_sha256",
        "_tree_sha256",
        "_open_exact_dir",
        "_read_anchored_child",
        "data_input_count",
        "source_hashes",
    ):
        assert marker in source


def test_independent_smoke() -> None:
    assert audit.smoke()["verdict"].startswith("GO_SYNTHETIC")


def test_independent_audit_recomputes_synthetic_artifacts(tmp_path: Path) -> None:
    (
        _plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        result = audit.audit_artifacts(
            evaluation,
            audit_root,
            plan_path,
            lock_path,
            attempt_dir,
        )
        # The independent auditor uses the canonical plan for schemas, while
        # counts and hashes are recomputed from the synthetic package.
        assert result["query_count"] == 4
        assert result["missing_truth_count"] == 1
    finally:
        for fd in fds.values():
            os.close(fd)


@pytest.mark.parametrize("root_kind", ["attempt", "evaluation", "audit"])
def test_independent_audit_rejects_symlink_roots(
    tmp_path: Path, root_kind: str
) -> None:
    (
        _plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        target = {
            "attempt": attempt_dir,
            "evaluation": evaluation,
            "audit": audit_root,
        }[root_kind]
        link = tmp_path / f"{root_kind}-root-link"
        link.symlink_to(target, target_is_directory=True)
        arguments = {
            "attempt": (
                evaluation,
                audit_root,
                plan_path,
                lock_path,
                link,
            ),
            "evaluation": (
                link,
                audit_root,
                plan_path,
                lock_path,
                attempt_dir,
            ),
            "audit": (
                evaluation,
                link,
                plan_path,
                lock_path,
                attempt_dir,
            ),
        }[root_kind]
        with pytest.raises(
            audit.IndependentAuditStopped, match="exact root"
        ):
            audit.audit_artifacts(*arguments)
    finally:
        for fd in fds.values():
            os.close(fd)


@pytest.mark.parametrize(
    ("root_kind", "child"),
    [
        ("attempt", "events.jsonl"),
        ("evaluation", "metrics.json"),
        ("audit", "provenance.json"),
    ],
)
def test_independent_audit_rejects_symlink_children(
    tmp_path: Path, root_kind: str, child: str
) -> None:
    (
        _plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        root = {
            "attempt": attempt_dir,
            "evaluation": evaluation,
            "audit": audit_root,
        }[root_kind]
        path = root / child
        saved = tmp_path / f"saved-{root_kind}-{child}"
        os.replace(path, saved)
        path.symlink_to(saved)
        with pytest.raises(
            audit.IndependentAuditStopped, match="anchored child"
        ):
            audit.audit_artifacts(
                evaluation, audit_root, plan_path, lock_path, attempt_dir
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_independent_audit_rejects_coordinated_ledger_attestation_reseal(
    tmp_path: Path,
) -> None:
    (
        plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        metrics_path = evaluation / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["verdict"] = "FORGED_COORDINATED_PACKAGE"
        metrics_path.write_bytes(evaluator.canonical_json(metrics))
        integrity_path = evaluation / "integrity.json"
        integrity = json.loads(integrity_path.read_text())
        integrity["metrics_sha256"] = hashlib.sha256(
            metrics_path.read_bytes()
        ).hexdigest()
        integrity_path.write_bytes(evaluator.canonical_json(integrity))
        evaluation_manifest_path = evaluation / "manifest.json"
        evaluation_manifest = json.loads(
            evaluation_manifest_path.read_text()
        )
        evaluation_manifest["files"]["metrics.json"] = (
            evaluator._json_record(metrics_path)
        )
        evaluation_manifest["files"]["integrity.json"] = (
            evaluator._json_record(integrity_path)
        )
        evaluation_manifest_path.write_bytes(
            evaluator.canonical_json(evaluation_manifest)
        )
        ledger_path = audit_root / "open_ledger.parquet"
        ledger_schema = pq.read_schema(ledger_path)
        rows = pq.read_table(ledger_path).to_pylist()
        forged_role = rows[0]["role"]
        rows[0]["size_bytes_before"] += 7
        rows[0]["size_bytes_after"] += 7
        rows[0]["sha256_before"] = "f" * 64
        rows[0]["sha256_after"] = "f" * 64
        pq.write_table(
            pa.Table.from_pylist(rows, schema=ledger_schema),
            ledger_path,
            compression="zstd",
        )
        audit_manifest_path = audit_root / "manifest.json"
        audit_manifest = json.loads(audit_manifest_path.read_text())
        provenance_path = audit_root / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["evaluation_manifest_sha256"] = hashlib.sha256(
            evaluation_manifest_path.read_bytes()
        ).hexdigest()
        provenance_path.write_bytes(evaluator.canonical_json(provenance))
        audit_manifest["files"]["open_ledger.parquet"] = (
            evaluator._json_record(ledger_path)
        )
        audit_manifest["files"]["provenance.json"] = evaluator._json_record(
            provenance_path
        )
        audit_manifest_path.write_bytes(
            evaluator.canonical_json(audit_manifest)
        )
        attestation_path = attempt_dir / "computed_attestation.json"
        attestation = json.loads(attestation_path.read_text())
        forged = attestation["input_snapshots"][forged_role]
        forged["size_bytes_before"] += 7
        forged["size_bytes_after"] += 7
        forged["sha256_before"] = "f" * 64
        forged["sha256_after"] = "f" * 64
        attestation["evaluation_manifest_sha256"] = hashlib.sha256(
            evaluation_manifest_path.read_bytes()
        ).hexdigest()
        attestation["evaluation_tree_sha256"] = evaluator._tree_sha256(
            evaluation, plan["outputs"]["runtime_files"]
        )
        attestation["audit_manifest_sha256"] = hashlib.sha256(
            audit_manifest_path.read_bytes()
        ).hexdigest()
        attestation["audit_tree_sha256"] = evaluator._tree_sha256(
            audit_root, plan["outputs"]["audit_files"]
        )
        _rewrite_attestation_and_authorization(
            plan, attempt_dir, attestation
        )
        with pytest.raises(
            audit.IndependentAuditStopped, match="live input mismatch"
        ):
            audit.audit_artifacts(
                evaluation, audit_root, plan_path, lock_path, attempt_dir
            )
    finally:
        for fd in fds.values():
            os.close(fd)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("measurement_slot_id", "f" * 64),
        ("attempt_id", "f" * 64),
    ],
)
def test_independent_audit_rejects_attestation_from_other_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    (
        plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        path = attempt_dir / "computed_attestation.json"
        attestation = json.loads(path.read_text())
        attestation[field] = value
        _rewrite_attestation_and_authorization(
            plan, attempt_dir, attestation
        )
        with pytest.raises(
            audit.IndependentAuditStopped, match="identity/hash mismatch"
        ):
            audit.audit_artifacts(
                evaluation, audit_root, plan_path, lock_path, attempt_dir
            )
    finally:
        for fd in fds.values():
            os.close(fd)


@pytest.mark.parametrize("payload", [b"{", b'{"substituted":true}\n'])
def test_independent_audit_rejects_truncated_or_substituted_attestation(
    tmp_path: Path, payload: bytes
) -> None:
    (
        _plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        path = attempt_dir / "computed_attestation.json"
        os.chmod(path, 0o600)
        path.write_bytes(payload)
        with pytest.raises(
            audit.IndependentAuditStopped,
            match="attestation hash differs",
        ):
            audit.audit_artifacts(
                evaluation, audit_root, plan_path, lock_path, attempt_dir
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_independent_audit_rejects_metrics_tamper(tmp_path: Path) -> None:
    (
        _plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        metrics_path = evaluation / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["gates"]["coverage_observed"] = 0.999
        metrics_path.write_bytes(evaluator.canonical_json(metrics))
        with pytest.raises(audit.IndependentAuditStopped):
            audit.audit_artifacts(
                evaluation,
                audit_root,
                plan_path,
                lock_path,
                attempt_dir,
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_independent_audit_rejects_rehashed_self_consistent_tamper(
    tmp_path: Path,
) -> None:
    (
        plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        metrics_path = evaluation / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["v412_measurements"][0]["coverage"] = evaluator._proportion(
            2, 4, plan["evaluation_spec"]["confidence_interval"]
        )
        metrics_path.write_bytes(evaluator.canonical_json(metrics))
        fixtures._reseal_self_consistent_hashes(evaluation, audit_root)
        with pytest.raises(audit.IndependentAuditStopped):
            audit.audit_artifacts(
                evaluation, audit_root, plan_path, lock_path, attempt_dir
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_independent_audit_rejects_fake_ledger_sizes_against_exact_inputs(
    tmp_path: Path,
) -> None:
    (
        _plan,
        _worker_spec,
        fds,
        evaluation,
        audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        assert audit.audit_artifacts(
            evaluation, audit_root, plan_path, lock_path, attempt_dir
        )["query_count"] == 4
        ledger_path = audit_root / "open_ledger.parquet"
        ledger_schema = pq.read_schema(ledger_path)
        rows = pq.read_table(ledger_path).to_pylist()
        rows[0]["size_bytes_before"] += 1
        rows[0]["size_bytes_after"] += 1
        pq.write_table(
            pa.Table.from_pylist(rows, schema=ledger_schema),
            ledger_path,
            compression="zstd",
        )
        manifest_path = audit_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["open_ledger.parquet"] = evaluator._json_record(
            ledger_path
        )
        manifest_path.write_bytes(evaluator.canonical_json(manifest))
        with pytest.raises(
            audit.IndependentAuditStopped,
            match="attestation final tree|ledger binding",
        ):
            audit.audit_artifacts(
                evaluation, audit_root, plan_path, lock_path, attempt_dir
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_independent_event_audit_detects_chain_tamper(tmp_path: Path) -> None:
    plan, lock = fixtures._attempt_plan_lock(tmp_path)
    plan_path = tmp_path / "synthetic-event-plan.json"
    plan_path.write_bytes(evaluator.canonical_json(plan))
    lock_path = tmp_path / "synthetic-event-lock.json"
    lock_path.write_bytes(evaluator.canonical_json(lock))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    attempt_dir, _slot, _attempt = evaluator.ensure_receipt(
        plan, lock, plan_sha, lock_sha, now=lambda: "t0"
    )
    result = audit.audit_event_journal(attempt_dir, plan_path, lock_path)
    assert result["state"] == "STARTED"
    events = attempt_dir / "events.jsonl"
    events.write_bytes(events.read_bytes() + b'{"partial":')
    with pytest.raises(audit.IndependentAuditStopped):
        audit.audit_event_journal(attempt_dir, plan_path, lock_path)


def test_independent_audit_rejects_self_consistent_fake_receipt_lock_hash(
    tmp_path: Path,
) -> None:
    (
        plan,
        _worker_spec,
        fds,
        _evaluation,
        _audit_root,
        plan_path,
        lock_path,
        attempt_dir,
    ) = _authorized_artifacts(tmp_path)
    try:
        receipt_path = attempt_dir / "receipt.json"
        os.chmod(receipt_path, 0o600)
        receipt = json.loads(receipt_path.read_text())
        receipt["lock_sha256"] = "f" * 64
        attempt_payload = {
            "schema_version": plan["attempt_protocol"]["schema_versions"][
                "attempt_identity"
            ],
            "plan_sha256": receipt["plan_sha256"],
            "lock_sha256": receipt["lock_sha256"],
            "input_hashes": receipt["input_hashes"],
            "evaluation_spec_sha256": receipt["evaluation_spec_sha256"],
            "worker_build_id": receipt["worker_build_id"],
            "oracle_build_id": receipt["oracle_build_id"],
            "parity_build_id": receipt["parity_build_id"],
        }
        receipt["attempt_id"] = hashlib.sha256(
            evaluator.canonical_json(attempt_payload)
        ).hexdigest()
        receipt_path.write_bytes(evaluator.canonical_json(receipt))
        events_path = attempt_dir / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text().splitlines()
        ]
        previous = None
        rebuilt = bytearray()
        for event in events:
            event["attempt_id"] = receipt["attempt_id"]
            event["previous_event_sha256"] = previous
            line = evaluator.canonical_json(event)
            rebuilt.extend(line)
            previous = hashlib.sha256(line).hexdigest()
        os.chmod(events_path, 0o600)
        events_path.write_bytes(bytes(rebuilt))
        state_path = attempt_dir / "state.json"
        os.chmod(state_path, 0o600)
        state = json.loads(state_path.read_text())
        state["attempt_id"] = receipt["attempt_id"]
        state_path.write_bytes(evaluator.canonical_json(state))
        with pytest.raises(
            audit.IndependentAuditStopped, match="receipt is not bound"
        ):
            audit.audit_event_journal(attempt_dir, plan_path, lock_path)
    finally:
        for fd in fds.values():
            os.close(fd)


def test_cli_requires_explicit_safe_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert audit.main([]) == 2
    assert "explicit" in capsys.readouterr().err
    assert audit.main(["--smoke"]) == 0
