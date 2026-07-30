#!/usr/bin/env python3
"""Run the disposable V4.12 successor identity gate on the external SSD.

This is not an authoritative S0 run.  It uses a distinct GATE domain and a
fresh temporary root, never reads real CRM data, and cannot authorize R3.
Its only purpose is to exercise the real worker ``_process`` under the same
private Python and deny-default Seatbelt boundary before R3 is preregistered.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts import (
        build_v412_fresh_intake_synthetic_fixture as core_builder,
    )
    from scripts import run_v412_fresh_s0_worker as worker
    from scripts import (
        seal_v412_fresh_intake_synthetic_execution_lock as sealer,
    )
except ModuleNotFoundError:
    import build_v412_fresh_intake_synthetic_fixture as core_builder
    import run_v412_fresh_s0_worker as worker
    import seal_v412_fresh_intake_synthetic_execution_lock as sealer


GATE_PARENT = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
GATE_PREFIX = "diag-r3-successor-gate."
GATE_RUN_DOMAIN = "SIRETO-V412-FRESH-SYNTHETIC-S0-GATE-RUN-ID\0"
GATE_PREDECESSOR_SHA256 = hashlib.sha256(
    b"SIRETO V4.12 DISPOSABLE SUCCESSOR GATE"
).hexdigest()
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class GateStop(RuntimeError):
    """Fail-closed disposable gate error."""


def _stop(message: str) -> None:
    raise GateStop(f"STOP {message}")


def _canonical(value: Any, *, final_lf: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def _write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    path.chmod(mode)


def _gate_inputs(
    plan: Mapping[str, Any], plan_raw: bytes
) -> tuple[str, str, dict[str, Any], dict[str, bytes]]:
    run_values = {
        "fixture_spec_sha256": plan["control_manifest"][
            "fixture_spec_sha256"
        ],
        "core_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "predecessor_receipt_sha256": GATE_PREDECESSOR_SHA256,
    }
    run_id = worker.core.opaque_digest(GATE_RUN_DOMAIN, run_values)
    csv_bytes = plan["fixture"]["csv"]["exact_utf8_text"].encode("utf-8")
    evidence_bytes = core_builder._empty_evidence_parquet(plan)
    source_payloads = core_builder._manifest_objects(
        plan, run_id, csv_bytes, evidence_bytes
    )
    control = {
        "schema_version": plan["control_manifest"]["schema"],
        "synthetic_fixture": True,
        "fixture_spec_sha256": plan["control_manifest"][
            "fixture_spec_sha256"
        ],
        "synthetic_run_id": run_id,
        "logical_time_utc": plan["fixture"]["logical_time_utc"],
        "batch_count": 1,
        "expected_source_row_count": 6,
        "producer_exclusions": [],
        "collection_source_manifest_sha256": hashlib.sha256(
            source_payloads["collection_source_manifest.json"]
        ).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            source_payloads["source_manifest.json"]
        ).hexdigest(),
        "crm_safe_csv_sha256": hashlib.sha256(
            source_payloads["crm_safe.csv"]
        ).hexdigest(),
        "evidence_source_manifest_sha256": hashlib.sha256(
            source_payloads["evidence_source_manifest.json"]
        ).hexdigest(),
        "evidence_source_parquet_sha256": hashlib.sha256(
            source_payloads["evidence_source.parquet"]
        ).hexdigest(),
    }
    control_raw = _canonical(control)
    attempt_values = {
        "synthetic_run_id": run_id,
        "fixture_control_manifest_sha256": hashlib.sha256(
            control_raw
        ).hexdigest(),
        "logical_time_utc": control["logical_time_utc"],
    }
    attempt_domain = plan["ids"]["attempt"]["domain"]
    attempt_id = worker.core.opaque_digest(attempt_domain, attempt_values)
    identity = {
        "schema_version": worker.SUCCESSOR_IDENTITY_SCHEMA,
        "algorithm": worker.SUCCESSOR_IDENTITY_ALGORITHM,
        "run": {
            "domain": GATE_RUN_DOMAIN,
            "projection": list(worker.SUCCESSOR_RUN_PROJECTION),
            "values": run_values,
            "result": run_id,
        },
        "attempt": {
            "domain": attempt_domain,
            "projection": list(worker.SUCCESSOR_ATTEMPT_PROJECTION),
            "values": attempt_values,
            "result": attempt_id,
        },
    }
    return (
        run_id,
        attempt_id,
        identity,
        {
            "CONTROL_MANIFEST": control_raw,
            **{
                role: source_payloads[name]
                for role, name in worker.PAYLOAD_NAMES.items()
            },
        },
    )


def _private_environment(
    runtime_root: Path, root: Path, run_id: str
) -> dict[str, str]:
    rootfs = runtime_root / "rootfs"
    pythonhome = (
        rootfs
        / "opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
        "Python.framework/Versions/3.14"
    )
    private_site = (
        rootfs / "opt/homebrew/lib/python3.14/site-packages"
    )
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHOME": str(pythonhome),
        "PYTHONPATH": f"{private_site}:{runtime_root / 'app'}",
        "TMPDIR": str(root / "tmp" / run_id),
    }


CHILD_CODE = r"""
import errno
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from scripts import run_v412_fresh_s0_worker as worker

def read_fd(fd):
    info = os.fstat(fd)
    return os.pread(fd, info.st_size, 0)

spec = json.loads(read_fd(int(sys.argv[1])))
result_fd = int(sys.argv[2])
payloads = {
    role: read_fd(fd) for role, fd in spec["payload_fds"].items()
}
worker_spec = {
    "synthetic_run_id": spec["synthetic_run_id"],
    "attempt_id": spec["attempt_id"],
    "logical_time_utc": spec["logical_time_utc"],
    "write_directory_fds": spec["write_directory_fds"],
}
outcome = None
exit_code = 2
try:
    denied = []
    for target in spec["file_canaries"]:
        try:
            fd = os.open(target, os.O_RDONLY)
        except OSError as exc:
            if exc.errno not in {errno.EPERM, errno.EACCES}:
                raise RuntimeError("file canary returned unexpected errno")
            denied.append(exc.errno)
        else:
            os.close(fd)
            raise RuntimeError("file canary opened")
    try:
        iterator = os.scandir(spec["parent_canary"])
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES}:
            raise RuntimeError("parent canary returned unexpected errno")
        denied.append(exc.errno)
    else:
        iterator.close()
        raise RuntimeError("parent canary enumerated")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect(("127.0.0.1", 9))
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES}:
            raise RuntimeError("network canary returned unexpected errno")
        denied.append(exc.errno)
    else:
        raise RuntimeError("network canary connected")
    finally:
        probe.close()
    try:
        fd = os.open(
            spec["write_parent_canary"],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES}:
            raise RuntimeError("write canary returned unexpected errno")
        denied.append(exc.errno)
    else:
        os.close(fd)
        raise RuntimeError("write parent canary created")
    first = {
        role: (
            os.fstat(fd).st_dev,
            os.fstat(fd).st_ino,
            os.fstat(fd).st_size,
            hashlib.sha256(read_fd(fd)).hexdigest(),
        )
        for role, fd in spec["payload_fds"].items()
    }
    started = time.monotonic()
    time.sleep(60.0)
    elapsed = time.monotonic() - started
    second = {
        role: (
            os.fstat(fd).st_dev,
            os.fstat(fd).st_ino,
            os.fstat(fd).st_size,
            hashlib.sha256(read_fd(fd)).hexdigest(),
        )
        for role, fd in spec["payload_fds"].items()
    }
    if elapsed < 60.0 or first != second:
        raise RuntimeError("stability failed")
    terminal, authority = worker._process(
        worker_spec,
        payloads,
        execution_identity=spec["execution_identity"],
        allowed_root=Path(spec["allowed_root"]),
    )
    outcome = {
        "schema_version": "sireto-v4.12-fresh-s0-successor-gate-result-1",
        "status": "GO",
        "terminal_result": terminal,
        "output_authority": authority,
        "canary_denied_count": len(denied),
        "canary_errnos": denied,
        "same_payload_fds": True,
        "monotonic_elapsed_seconds": f"{elapsed:.9f}",
    }
    exit_code = 0
except worker.WorkerStop as exc:
    outcome = {
        "schema_version": "sireto-v4.12-fresh-s0-successor-gate-result-1",
        "status": "STOP",
        "worker_phase": exc.worker_phase,
        "worker_reason_code": exc.worker_reason_code,
    }
except Exception:
    outcome = {
        "schema_version": "sireto-v4.12-fresh-s0-successor-gate-result-1",
        "status": "STOP",
        "worker_phase": "WORKER_RUNTIME",
        "worker_reason_code": "INTERNAL_ERROR",
    }
raw = json.dumps(
    outcome,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8") + b"\n"
os.write(result_fd, raw)
raise SystemExit(exit_code)
"""

NEGATIVE_CHILD_CODE = r"""
import json
import os
import sys
from pathlib import Path
from scripts import run_v412_fresh_s0_worker as worker

def read_fd(fd):
    info = os.fstat(fd)
    return os.pread(fd, info.st_size, 0)

spec = json.loads(read_fd(int(sys.argv[1])))
result_fd = int(sys.argv[2])
payloads = {
    role: read_fd(fd) for role, fd in spec["payload_fds"].items()
}
worker_spec = {
    "synthetic_run_id": spec["synthetic_run_id"],
    "attempt_id": spec["attempt_id"],
    "logical_time_utc": spec["logical_time_utc"],
    "write_directory_fds": spec["write_directory_fds"],
}
try:
    worker._process(
        worker_spec,
        payloads,
        execution_identity=spec["execution_identity"],
        allowed_root=Path(spec["allowed_root"]),
    )
    outcome = {
        "schema_version": "sireto-v4.12-fresh-s0-successor-gate-negative-1",
        "status": "UNEXPECTED_SUCCESS",
    }
    exit_code = 0
except worker.WorkerStop as exc:
    outcome = {
        "schema_version": "sireto-v4.12-fresh-s0-successor-gate-negative-1",
        "status": "STOP",
        "worker_phase": exc.worker_phase,
        "worker_reason_code": exc.worker_reason_code,
    }
    exit_code = 2
except Exception:
    outcome = {
        "schema_version": "sireto-v4.12-fresh-s0-successor-gate-negative-1",
        "status": "STOP",
        "worker_phase": "WORKER_RUNTIME",
        "worker_reason_code": "INTERNAL_ERROR",
    }
    exit_code = 2
raw = json.dumps(
    outcome,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8") + b"\n"
os.write(result_fd, raw)
raise SystemExit(exit_code)
"""


def _tree_snapshot(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, root in sorted(paths.items()):
        for path in sorted(
            root.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")
        ):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                records.append(
                    {"role": role, "path": relative, "kind": "DIRECTORY"}
                )
            elif path.is_file():
                payload = path.read_bytes()
                records.append(
                    {
                        "role": role,
                        "path": relative,
                        "kind": "FILE",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            else:
                _stop("gate output tree contains a special entry")
    return records


def run_gate() -> dict[str, Any]:
    if not GATE_PARENT.is_dir():
        _stop("external SSD gate parent is absent")
    root = Path(tempfile.mkdtemp(prefix=GATE_PREFIX, dir=GATE_PARENT))
    root.chmod(0o700)
    launch_plan, core_plan = sealer._load_plans()
    core_plan_raw = (
        sealer.REPOSITORY / launch_plan["core"]["pins"]["plan"]["path"]
    ).read_bytes()
    run_id, attempt_id, identity, payloads = _gate_inputs(
        core_plan, core_plan_raw
    )

    package = root / "inbox" / run_id / "package"
    control_dir = root / "control" / run_id
    for path in (package, control_dir):
        path.mkdir(parents=True, mode=0o700)
        path.chmod(0o700)
    payload_paths: dict[str, Path] = {}
    for role, payload in payloads.items():
        name = (
            "fixture_control_manifest.json"
            if role == "CONTROL_MANIFEST"
            else worker.PAYLOAD_NAMES[role]
        )
        path = control_dir / name if role == "CONTROL_MANIFEST" else package / name
        _write(path, payload, 0o400)
        payload_paths[role] = path

    write_paths = {
        "SEALED": root / "sealed" / run_id,
        "SCAN": root / "scan" / run_id,
        "QUARANTINE": root / "quarantine" / run_id,
        "AUDIT": root / "audit" / run_id / "worker",
        "TMP": root / "tmp" / run_id,
    }
    for path in write_paths.values():
        path.mkdir(parents=True, mode=0o700)
        path.chmod(0o700)
    audit_parent = root / "audit" / run_id / "parent"
    audit_parent.mkdir(mode=0o700)
    audit_parent.chmod(0o700)

    implementation_payloads = {
        "WORKER": (
            sealer.REPOSITORY / "scripts/run_v412_fresh_s0_worker.py"
        ).read_bytes(),
        "SANDBOX_PROFILE": (
            sealer.REPOSITORY
            / "config/v4_12_fresh_intake_synthetic_scanner_sealer.sb"
        ).read_bytes(),
    }
    (
        runtime_manifest,
        _manifest_path,
        profile_sha,
        private_python,
        profile_path,
    ) = sealer._build_private_runtime(
        launch_plan,
        "0" * 40,
        implementation_payloads,
        root,
        run_id,
    )
    _canary_manifest, _canary_path = sealer._create_canaries(
        launch_plan, root, run_id
    )

    payload_fds: dict[str, int] = {}
    write_fds: dict[str, int] = {}
    opened: list[int] = []
    result_read, result_write = os.pipe()
    opened.extend((result_read, result_write))
    try:
        for role in worker.PAYLOAD_ROLES:
            fd = os.open(payload_paths[role], os.O_RDONLY)
            payload_fds[role] = fd
            opened.append(fd)
        for role, path in write_paths.items():
            fd = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            write_fds[role] = fd
            opened.append(fd)
        spec = {
            "allowed_root": str(root),
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "logical_time_utc": core_plan["fixture"]["logical_time_utc"],
            "execution_identity": identity,
            "payload_fds": payload_fds,
            "write_directory_fds": write_fds,
            "file_canaries": [
                str(root / "canaries/forbidden/data/sentinel"),
                str(root / "canaries/forbidden/models/sentinel"),
                str(root / "canaries/forbidden/reports/sentinel"),
                str(root / "canaries/forbidden/challenges/sentinel"),
                str(
                    root
                    / "canaries/forbidden/final_holdout_inputs/sentinel"
                ),
                str(
                    root / "canaries/forbidden/fresh_holdout_intake/sentinel"
                ),
                str(
                    root
                    / "canaries/forbidden/"
                    "fresh_holdout_evaluation_ledger/sentinel"
                ),
                str(root / "canaries/forbidden/registries/sentinel"),
            ],
            "parent_canary": str(root / "canaries/forbidden"),
            "write_parent_canary": str(
                root
                / "audit"
                / run_id
                / "parent"
                / "worker-write-sentinel"
            ),
        }
        spec_path = control_dir / "gate_spec.json"
        _write(spec_path, _canonical(spec), 0o400)
        spec_fd = os.open(spec_path, os.O_RDONLY)
        opened.append(spec_fd)
        profile_raw = profile_path.read_bytes()
        argv = [
            str(sealer.SANDBOX_EXEC),
            "-p",
            profile_raw.decode("utf-8"),
            str(private_python),
            "-c",
            CHILD_CODE,
            str(spec_fd),
            str(result_write),
        ]
        pass_fds = [
            spec_fd,
            result_write,
            *payload_fds.values(),
            *write_fds.values(),
        ]
        return_code, stdout, stderr = sealer._run_bounded_child(
            argv,
            _private_environment(root / "runtime" / run_id, root, run_id),
            timeout_seconds=150,
            capture_limit_bytes_each=65_536,
            pass_fds=pass_fds,
        )
        os.close(result_write)
        opened.remove(result_write)
        result_raw = b""
        while True:
            chunk = os.read(result_read, 65_536)
            if not chunk:
                break
            result_raw += chunk
        if stdout or stderr:
            _stop("sandbox gate wrote stdout or stderr")
        try:
            result = json.loads(result_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _stop(f"sandbox gate result is invalid: {exc}")
        if result_raw != _canonical(result):
            _stop("sandbox gate result is not canonical")
        if (
            return_code != 0
            or result.get("status") != "GO"
            or result.get("terminal_result")
            != "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"
            or result.get("canary_denied_count") != 11
            or len(result.get("canary_errnos", [])) != 11
            or any(
                value not in {1, 13}
                for value in result.get("canary_errnos", [])
            )
            or result.get("same_payload_fds") is not True
            or result.get("output_authority", {}).get(
                "terminal_tree_kind"
            )
            != "SCAN_OUTPUT"
            or result.get("output_authority", {}).get(
                "journal_generation"
            )
            != 3
        ):
            _stop("sandbox gate did not reach the closed GO result")
        before_negative = _tree_snapshot(write_paths)
        negative_spec = json.loads(_canonical(spec))
        r1_values = {
            "fixture_spec_sha256": core_plan["control_manifest"][
                "fixture_spec_sha256"
            ],
            "plan_sha256": hashlib.sha256(core_plan_raw).hexdigest(),
        }
        negative_spec["execution_identity"]["run"] = {
            "domain": core_plan["ids"]["run"]["domain"],
            "projection": ["fixture_spec_sha256", "plan_sha256"],
            "values": r1_values,
            "result": worker.core.opaque_digest(
                core_plan["ids"]["run"]["domain"], r1_values
            ),
        }
        negative_spec_path = control_dir / "gate_spec_r1_reject.json"
        _write(negative_spec_path, _canonical(negative_spec), 0o400)
        negative_spec_fd = os.open(negative_spec_path, os.O_RDONLY)
        opened.append(negative_spec_fd)
        negative_read, negative_write = os.pipe()
        opened.extend((negative_read, negative_write))
        negative_argv = [
            str(sealer.SANDBOX_EXEC),
            "-p",
            profile_raw.decode("utf-8"),
            str(private_python),
            "-c",
            NEGATIVE_CHILD_CODE,
            str(negative_spec_fd),
            str(negative_write),
        ]
        negative_code, negative_stdout, negative_stderr = (
            sealer._run_bounded_child(
                negative_argv,
                _private_environment(
                    root / "runtime" / run_id, root, run_id
                ),
                timeout_seconds=30,
                capture_limit_bytes_each=65_536,
                pass_fds=[
                    negative_spec_fd,
                    negative_write,
                    *payload_fds.values(),
                    *write_fds.values(),
                ],
            )
        )
        os.close(negative_write)
        opened.remove(negative_write)
        negative_raw = b""
        while True:
            chunk = os.read(negative_read, 65_536)
            if not chunk:
                break
            negative_raw += chunk
        negative = json.loads(negative_raw)
        if (
            negative_code != 2
            or negative_stdout
            or negative_stderr
            or negative_raw != _canonical(negative)
            or negative
            != {
                "schema_version": (
                    "sireto-v4.12-fresh-s0-successor-gate-negative-1"
                ),
                "status": "STOP",
                "worker_phase": "IDENTITY",
                "worker_reason_code": "EXECUTION_IDENTITY_SCHEMA_INVALID",
            }
            or _tree_snapshot(write_paths) != before_negative
        ):
            _stop("sandbox gate did not reject R1 without output mutation")
        result.update(
            {
                "gate_root": str(root),
                "synthetic_run_id": run_id,
                "attempt_id": attempt_id,
                "worker_sha256": hashlib.sha256(
                    implementation_payloads["WORKER"]
                ).hexdigest(),
                "effective_profile_sha256": profile_sha,
                "runtime_dependency_closure_sha256": runtime_manifest[
                    "dependency_closure_sha256"
                ],
                "stdout_sha256": EMPTY_SHA256,
                "stderr_sha256": EMPTY_SHA256,
                "r1_rejection": negative,
                "r1_rejection_output_tree_unchanged": True,
            }
        )
        gate_result_path = (
            root
            / "audit"
            / run_id
            / "parent"
            / "gate_results"
            / "successor_gate_result.json"
        )
        result["gate_result_path"] = str(gate_result_path)
        _write(gate_result_path, _canonical(result), 0o400)
        return result
    finally:
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass


def main() -> int:
    if len(sys.argv) != 1:
        print("STOP successor gate accepts no arguments", file=sys.stderr)
        return 2
    try:
        result = run_gate()
    except GateStop as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(_canonical(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
