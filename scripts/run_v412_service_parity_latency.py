#!/usr/bin/env python3
"""Parent-owned, paired persistent V4.11/V4.12-G service gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any

REPOSITORY = Path("/Users/nathanjullia/Documents/Projets/SIRETO")
sys.path.insert(0, str(REPOSITORY))

from src.xgb_matcher.v412_service_execution_lock import (  # noqa: E402
    OUTPUT_ROOT,
    validate_execution_lock,
)
from src.xgb_matcher.v412_service_parity import (  # noqa: E402
    _capture_untrusted_regular,
    evaluate_paired_gate,
    validate_worker_output,
)


STOP = "STOP_V412_SERVICE_INTEGRITY"
BOOTSTRAP = REPOSITORY / "scripts/run_v412_persistent_service_bootstrap.py"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_new(path: Path, payload: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ValueError(f"{STOP}: short parent write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = _capture_untrusted_regular(path)
    if observed != payload:
        raise ValueError(f"{STOP}: parent output reseal mismatch")
    return hashlib.sha256(payload).hexdigest()


def _worker_nonce(parent_nonce: str, phase: str, mode: str) -> str:
    return hashlib.sha256(
        f"{parent_nonce}:{phase}:{mode}".encode("ascii")
    ).hexdigest()


def _launch_worker(
    *,
    staging: Path,
    phase: str,
    mode: str,
    parent_nonce: str,
    execution_lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    child_nonce = _worker_nonce(parent_nonce, phase, mode)
    output = staging / f"{phase}_{mode}"
    command = [
        sys.executable,
        str(BOOTSTRAP),
        "--mode",
        mode,
        "--phase",
        phase,
        "--run-nonce",
        child_nonce,
        "--execution-lock-sha256",
        execution_lock_sha256,
        "--output-dir",
        str(output),
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
            "SIRETO_NETWORK_AUDIT_DENY",
        }
    }
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    launch = {
        "phase": phase,
        "mode": mode,
        "pid": process.pid,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "child_nonce": child_nonce,
        "output": output.name,
    }
    if process.returncode != 0:
        raise ValueError(
            f"{STOP}: worker {phase}/{mode} exited "
            f"{process.returncode}: {stderr[-2000:]}"
        )
    summary = validate_worker_output(
        output,
        expected_mode=mode,
        expected_phase=phase,
        expected_nonce=child_nonce,
        expected_pid=process.pid,
        expected_execution_lock_sha256=execution_lock_sha256,
    )
    manifest_payload = _capture_untrusted_regular(output / "manifest.json")
    launch["manifest_sha256"] = hashlib.sha256(
        manifest_payload
    ).hexdigest()
    return summary, launch


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if not key.startswith("_")
    }


def run() -> tuple[str, Path]:
    lock, lock_sha256 = validate_execution_lock(verify_git=True)
    OUTPUT_ROOT.mkdir(mode=0o700, parents=False, exist_ok=True)
    parent_nonce = secrets.token_hex(32)
    staging = OUTPUT_ROOT / f".staging-{parent_nonce}"
    staging.mkdir(mode=0o700)
    launches: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for phase in ("diagnostic", "gate"):
            for mode in ("v411", "v412g"):
                summary, launch = _launch_worker(
                    staging=staging,
                    phase=phase,
                    mode=mode,
                    parent_nonce=parent_nonce,
                    execution_lock_sha256=lock_sha256,
                )
                summaries[f"{phase}_{mode}"] = summary
                launches.append(launch)
        pids = [record["pid"] for record in launches]
        if len(set(pids)) != 4:
            raise ValueError(f"{STOP}: workers did not use four processes")
        gate = evaluate_paired_gate(
            summaries["gate_v411"],
            summaries["gate_v412g"],
        )
        report = {
            "schema_version": "sireto-v4.12-service-parity-gate-1",
            "verdict": gate["verdict"],
            "execution_lock_sha256": lock_sha256,
            "locked_commit": lock["git_commit"],
            "parent_pid": os.getpid(),
            "parent_nonce": parent_nonce,
            "order": [
                "diagnostic_v411",
                "diagnostic_v412g",
                "gate_v411",
                "gate_v412g",
            ],
            "first_pass_diagnostic_only": True,
            "gate_uses_warm_filesystem": True,
            "launches": launches,
            "summaries": {
                name: _public_summary(summary)
                for name, summary in summaries.items()
            },
            "paired_gate": gate,
            "claims": {
                "precision_product_measured": False,
                "test_final_opened": False,
                "labels_opened": False,
                "models_retrained": False,
            },
        }
        report_sha256 = _write_new(
            staging / "report.json",
            _canonical_json(report),
        )
        seal = {
            "schema_version": "sireto-v4.12-service-parity-seal-1",
            "report_sha256": report_sha256,
            "verdict": gate["verdict"],
        }
        _write_new(staging / "seal.json", _canonical_json(seal))
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        final = OUTPUT_ROOT / parent_nonce[:16]
        os.rename(staging, final)
        root_fd = os.open(OUTPUT_ROOT, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        return gate["verdict"], final
    except BaseException as exc:
        failure = {
            "schema_version": "sireto-v4.12-service-parity-failure-1",
            "verdict": STOP,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "execution_lock_sha256": lock_sha256,
            "launches": launches,
        }
        try:
            _write_new(staging / "failure.json", _canonical_json(failure))
        except BaseException:
            pass
        raise


def main() -> int:
    try:
        verdict, output = run()
    except Exception as exc:
        print(f"{STOP}: {exc}", file=sys.stderr)
        return 65
    print(
        json.dumps(
            {"verdict": verdict, "output": str(output)},
            sort_keys=True,
        )
    )
    return 0 if verdict == "GO_V412_SERVICE_FREEZE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
