#!/usr/bin/env python3
"""Build the preregistered V4.12 synthetic S0-R3 fixture exactly once.

Only the pinned six-row core fixture is used.  The consumed R2 terminal
receipt is verified as an immutable predecessor authority before any output
root is created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
R3_PLAN_PATH = REPOSITORY_ROOT / "config/v4_12_fresh_s0_r3_plan.json"
R3_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/"
    "fresh_holdout_intake_synthetic_r3"
)
PREREG_PLAN_SHA256 = (
    "ce7f8ed4a9d6236e61cffca72b92a1043d414afc69571ae79c94f191e6def1e2"
)
EXPECTED_RUN_ID = (
    "kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl"
)
EXPECTED_ATTEMPT_ID = (
    "afjgbfncbfdbcakcjiclhmlnmgmemcjmllkhdfgogjjncompjojcnbkelopdklgp"
)
EXPECTED_CONTROL_SHA256 = (
    "bae57c4f207f2637574a2872169a311a9199f3fc6ba89c2694ddd123b245ac18"
)
IDENTITY_SCHEMA = "sireto-v4.12-fresh-s0-successor-identity-1"
IDENTITY_ALGORITHM = (
    "SHA256_DOMAIN_UTF8_CONCAT_CANONICAL_JSON_SORT_KEYS_COMPACT_UTF8_NO_LF_"
    "THEN_MAP_HEX_NIBBLES_0_TO_F_TO_ASCII_A_TO_P"
)
RUN_DOMAIN = "SIRETO-V412-FRESH-SYNTHETIC-S0-R3-RUN-ID\0"
RUN_PROJECTION = (
    "fixture_spec_sha256",
    "core_plan_sha256",
    "predecessor_receipt_sha256",
)
ATTEMPT_DOMAIN = "SIRETO-V412-FRESH-SYNTHETIC-ATTEMPT-ID\0"
ATTEMPT_PROJECTION = (
    "synthetic_run_id",
    "fixture_control_manifest_sha256",
    "logical_time_utc",
)
BASE_AUTHORITY_ROLES = (
    "core_contract",
    "core_plan",
    "r2_contract",
    "r2_plan",
)
CORE_PIN_ROLES = ("contract", "fixture_builder", "plan", "scanner", "tests")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_ID = re.compile(r"^[a-p]{64}$")


class R3FixtureBuildError(RuntimeError):
    """Fail-closed R3 fixture construction error."""


def _stop(message: str) -> None:
    raise R3FixtureBuildError(message)


def canonical_json_bytes(value: Any, *, final_lf: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def opaque_digest(domain: str, values: Mapping[str, Any]) -> str:
    digest = sha256_bytes(
        domain.encode("utf-8")
        + canonical_json_bytes(values, final_lf=False)
    )
    return "".join(chr(ord("a") + int(nibble, 16)) for nibble in digest)


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid constant {value}")

    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _stop(f"{label} is invalid JSON: {exc}")
    if type(value) is not dict or raw != canonical_json_bytes(value):
        _stop(f"{label} is not canonical JSON with one final LF")
    return value


def _normalized_absolute(path: Path, label: str) -> Path:
    raw = os.fspath(path)
    if not path.is_absolute() or "\0" in raw:
        _stop(f"{label} must be absolute")
    normalized = Path(os.path.abspath(raw))
    if path != normalized or any(part in {"", ".", ".."} for part in path.parts):
        _stop(f"{label} must be normalized")
    return normalized


def _read_regular_anchored(path: Path, label: str) -> bytes:
    normalized = _normalized_absolute(path, label)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open("/", directory_flags)
    file_fd: int | None = None
    try:
        for component in normalized.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child
        file_fd = os.open(normalized.name, file_flags, dir_fd=directory_fd)
    except OSError as exc:
        _stop(f"{label} cannot be opened safely: {exc}")
    finally:
        os.close(directory_fd)
    assert file_fd is not None
    try:
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            _stop(f"{label} identity or permissions are unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 65_536)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_uid",
            "st_nlink",
            "st_mode",
        )
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            _stop(f"{label} changed while being read")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            _stop(f"{label} changed size while being read")
        return raw
    finally:
        os.close(file_fd)


def _repo_path(value: Any, label: str) -> Path:
    if type(value) is not str:
        _stop(f"{label} path is not a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _stop(f"{label} repository path is invalid")
    result = REPOSITORY_ROOT / relative
    try:
        result.relative_to(REPOSITORY_ROOT)
    except ValueError:
        _stop(f"{label} escapes the repository")
    return result


def _pin(value: Any, label: str) -> tuple[Path, str]:
    if (
        type(value) is not dict
        or set(value) != {"path", "sha256"}
        or type(value["sha256"]) is not str
        or HEX64.fullmatch(value["sha256"]) is None
    ):
        _stop(f"{label} pin is invalid")
    return _repo_path(value["path"], label), value["sha256"]


def _read_pinned(value: Any, label: str) -> tuple[Path, bytes]:
    path, expected = _pin(value, label)
    raw = _read_regular_anchored(path, label)
    if sha256_bytes(raw) != expected:
        _stop(f"{label} hash mismatch")
    return path, raw


def _validate_identity(
    identity: Any,
    *,
    core_plan: Mapping[str, Any],
    core_plan_raw: bytes,
    predecessor_sha256: str,
) -> tuple[str, Mapping[str, Any]]:
    if (
        type(identity) is not dict
        or set(identity) != {"schema_version", "algorithm", "run", "attempt"}
        or identity["schema_version"] != IDENTITY_SCHEMA
        or identity["algorithm"] != IDENTITY_ALGORITHM
    ):
        _stop("R3 execution identity schema mismatch")
    run = identity["run"]
    attempt = identity["attempt"]
    for label, component in (("run", run), ("attempt", attempt)):
        if (
            type(component) is not dict
            or set(component) != {"domain", "projection", "values", "result"}
            or type(component["domain"]) is not str
            or type(component["projection"]) is not list
            or type(component["values"]) is not dict
            or type(component["result"]) is not str
            or OPAQUE_ID.fullmatch(component["result"]) is None
        ):
            _stop(f"R3 {label} identity component mismatch")
    expected_run_values = {
        "fixture_spec_sha256": core_plan["control_manifest"][
            "fixture_spec_sha256"
        ],
        "core_plan_sha256": sha256_bytes(core_plan_raw),
        "predecessor_receipt_sha256": predecessor_sha256,
    }
    if (
        run["domain"] != RUN_DOMAIN
        or tuple(run["projection"]) != RUN_PROJECTION
        or run["values"] != expected_run_values
        or run["result"] != opaque_digest(RUN_DOMAIN, expected_run_values)
        or run["result"] != EXPECTED_RUN_ID
    ):
        _stop("R3 run identity authority mismatch")
    if (
        attempt["domain"] != ATTEMPT_DOMAIN
        or tuple(attempt["projection"]) != ATTEMPT_PROJECTION
        or set(attempt["values"]) != set(ATTEMPT_PROJECTION)
        or attempt["values"]["synthetic_run_id"] != run["result"]
        or attempt["result"] != opaque_digest(
            ATTEMPT_DOMAIN, attempt["values"]
        )
        or attempt["result"] != EXPECTED_ATTEMPT_ID
    ):
        _stop("R3 attempt identity authority mismatch")
    return run["result"], attempt


def _verify_predecessor(
    predecessor: Any,
    *,
    r2_plan: Mapping[str, Any],
) -> bytes:
    exact_predecessor_fields = {
        "attempt_id",
        "reason_code",
        "receipt_path",
        "receipt_sha256",
        "run_id",
        "terminal_result",
        "verdict",
    }
    if (
        type(predecessor) is not dict
        or set(predecessor) != exact_predecessor_fields
        or type(predecessor["receipt_sha256"]) is not str
        or HEX64.fullmatch(predecessor["receipt_sha256"]) is None
        or type(predecessor["run_id"]) is not str
        or OPAQUE_ID.fullmatch(predecessor["run_id"]) is None
        or type(predecessor["attempt_id"]) is not str
        or OPAQUE_ID.fullmatch(predecessor["attempt_id"]) is None
    ):
        _stop("R2 predecessor authority schema mismatch")
    receipt_path = _normalized_absolute(
        Path(predecessor["receipt_path"]), "R2 predecessor receipt path"
    )
    raw = _read_regular_anchored(receipt_path, "R2 predecessor receipt")
    if sha256_bytes(raw) != predecessor["receipt_sha256"]:
        _stop("R2 predecessor receipt hash mismatch")
    receipt = _strict_json(raw, "R2 predecessor receipt")
    receipt_contract = r2_plan.get("launch_receipt")
    if (
        type(receipt_contract) is not dict
        or type(receipt_contract.get("exact_fields")) is not list
        or set(receipt) != set(receipt_contract["exact_fields"])
        or receipt.get("schema_version")
        != receipt_contract.get("schema_version")
    ):
        _stop("R2 predecessor receipt schema mismatch")
    expected = {
        "synthetic_run_id": predecessor["run_id"],
        "attempt_id": predecessor["attempt_id"],
        "phase": "RECEIPT",
        "reason_code": predecessor["reason_code"],
        "terminal_result": predecessor["terminal_result"],
        "verdict": predecessor["verdict"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        _stop("R2 predecessor receipt terminal authority mismatch")
    if (
        predecessor["verdict"] != "STOP"
        or predecessor["reason_code"] != "WORKER_CONTROLLED_STOP"
        or predecessor["terminal_result"]
        != "STOP_SYNTHETIC_SCANNER_SEALER_V412"
    ):
        _stop("R2 predecessor constants mismatch")
    worker = receipt.get("worker_receipt")
    worker_schema = r2_plan.get("schema_definitions", {}).get(
        "worker_receipt_record"
    )
    control_result = (
        worker.get("control_result") if type(worker) is dict else None
    )
    if (
        type(worker_schema) is not dict
        or type(worker) is not dict
        or set(worker) != set(worker_schema.get("exact_fields", ()))
        or worker.get("exit_code") != 2
        or worker.get("signal") is not None
        or worker.get("stdout_size_bytes") != 0
        or worker.get("stderr_size_bytes") != 0
        or type(control_result) is not dict
        or control_result.get("message_type") != "STOP"
        or control_result.get("reason_code")
        != "WORKER_CONTROLLED_STOP"
    ):
        _stop("R2 predecessor worker receipt mismatch")
    return raw


def _load_authorities(
    plan_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    ModuleType,
    bytes,
    bytes,
]:
    normalized_plan = _normalized_absolute(plan_path, "R3 plan path")
    plan_raw = _read_regular_anchored(normalized_plan, "R3 plan")
    if (
        normalized_plan == R3_PLAN_PATH
        and sha256_bytes(plan_raw) != PREREG_PLAN_SHA256
    ):
        _stop("production R3 plan differs from preregistration")
    plan = _strict_json(plan_raw, "R3 plan")
    expected_top = {
        "base_authorities",
        "canonical_json",
        "contract",
        "execution_identity",
        "future_implementation",
        "gate_authority",
        "implementation_baseline",
        "implementation_blob_roles",
        "inheritance",
        "inherited_authorities",
        "paths",
        "policy",
        "predecessor",
        "r3_successor",
        "required_implementation_gate",
        "schemas",
        "sequence",
        "source_blob_roles",
        "status",
        "verdicts",
    }
    if set(plan) != expected_top:
        _stop("R3 plan fields mismatch")
    _, contract_raw = _read_pinned(plan["contract"], "R3 contract")
    del contract_raw
    authorities = plan.get("base_authorities")
    if (
        type(authorities) is not dict
        or tuple(sorted(authorities)) != tuple(sorted(BASE_AUTHORITY_ROLES))
    ):
        _stop("R3 base authority role set mismatch")
    base_raw: dict[str, bytes] = {}
    for role in BASE_AUTHORITY_ROLES:
        _, base_raw[role] = _read_pinned(
            authorities[role], f"R3 base {role}"
        )
    core_plan = _strict_json(base_raw["core_plan"], "pinned core plan")
    r2_plan = _strict_json(base_raw["r2_plan"], "pinned R2 plan")
    core = r2_plan.get("core")
    pins = core.get("pins") if type(core) is dict else None
    if (
        type(core) is not dict
        or core.get("immutable") is not True
        or type(pins) is not dict
        or tuple(sorted(pins)) != tuple(sorted(CORE_PIN_ROLES))
    ):
        _stop("pinned R2 core authority mismatch")
    pinned: dict[str, bytes] = {}
    pinned_paths: dict[str, Path] = {}
    for role in CORE_PIN_ROLES:
        pinned_paths[role], pinned[role] = _read_pinned(
            pins[role], f"pinned core {role}"
        )
    if (
        pinned["plan"] != base_raw["core_plan"]
        or pinned["contract"] != base_raw["core_contract"]
        or core_plan.get("contract") != pins["contract"]
    ):
        _stop("R3 base and R2 core pins disagree")
    fixture_sha = sha256_bytes(
        canonical_json_bytes(core_plan["fixture"], final_lf=False)
    )
    if (
        fixture_sha
        != core_plan["control_manifest"]["fixture_spec_sha256"]
    ):
        _stop("pinned core fixture specification mismatch")
    module_spec = importlib.util.spec_from_file_location(
        "_sireto_v412_r3_pinned_core_builder",
        pinned_paths["fixture_builder"],
    )
    if module_spec is None or module_spec.loader is None:
        _stop("cannot load pinned core fixture builder")
    core_builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(core_builder)
    return (
        plan,
        core_plan,
        r2_plan,
        core_builder,
        base_raw["core_plan"],
        plan_raw,
    )


def _control(
    core_plan: Mapping[str, Any],
    run_id: str,
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    fixture = core_plan["fixture"]
    return {
        "schema_version": core_plan["control_manifest"]["schema"],
        "synthetic_fixture": True,
        "fixture_spec_sha256": core_plan["control_manifest"][
            "fixture_spec_sha256"
        ],
        "synthetic_run_id": run_id,
        "logical_time_utc": fixture["logical_time_utc"],
        "batch_count": 1,
        "expected_source_row_count": fixture["expected"]["source_row_count"],
        "producer_exclusions": [],
        "collection_source_manifest_sha256": sha256_bytes(
            payloads["collection_source_manifest.json"]
        ),
        "source_manifest_sha256": sha256_bytes(
            payloads["source_manifest.json"]
        ),
        "crm_safe_csv_sha256": sha256_bytes(payloads["crm_safe.csv"]),
        "evidence_source_manifest_sha256": sha256_bytes(
            payloads["evidence_source_manifest.json"]
        ),
        "evidence_source_parquet_sha256": sha256_bytes(
            payloads["evidence_source.parquet"]
        ),
    }


def _validate_root(
    root: Path, plan: Mapping[str, Any], core_builder: ModuleType
) -> Path:
    requested = _normalized_absolute(root, "requested R3 root")
    planned = _normalized_absolute(
        Path(plan["paths"]["allowed_root"]), "planned R3 root"
    )
    if (
        plan.get("r3_successor", {}).get("root") != str(planned)
        or requested != planned
    ):
        _stop("requested R3 root differs from the preregistered plan")
    if core_builder._validate_root(requested) != requested:
        _stop("core root validation changed the R3 root")
    return requested


def build_fixture(
    *,
    root: Path,
    plan_path: Path = R3_PLAN_PATH,
) -> dict[str, Any]:
    """Build an immutable R3 package from the preregistered authorities."""

    old_umask = os.umask(0o077)
    try:
        (
            plan,
            core_plan,
            r2_plan,
            core_builder,
            core_plan_raw,
            _plan_raw,
        ) = _load_authorities(Path(plan_path))
        root = _validate_root(root, plan, core_builder)
        predecessor_raw = _verify_predecessor(
            plan["predecessor"], r2_plan=r2_plan
        )
        run_id, attempt_authority = _validate_identity(
            plan["execution_identity"],
            core_plan=core_plan,
            core_plan_raw=core_plan_raw,
            predecessor_sha256=sha256_bytes(predecessor_raw),
        )
        if (
            plan["r3_successor"].get("execution_identity_source")
            != "execution_identity"
            or plan["r3_successor"].get("predecessor_source")
            != "predecessor"
            or plan["r3_successor"].get("single_attempt") is not True
        ):
            _stop("R3 successor authority mismatch")

        csv_bytes = core_plan["fixture"]["csv"]["exact_utf8_text"].encode(
            "utf-8"
        )
        evidence_bytes = core_builder._empty_evidence_parquet(core_plan)
        payloads = core_builder._manifest_objects(
            core_plan, run_id, csv_bytes, evidence_bytes
        )
        if set(payloads) != set(core_plan["input_package"]["exact_files"]):
            _stop("pinned core builder emitted an unexpected payload set")
        if payloads["crm_safe.csv"] != csv_bytes:
            _stop("pinned core builder changed the synthetic CSV")
        if payloads["evidence_source.parquet"] != evidence_bytes:
            _stop("pinned core builder changed the synthetic evidence")

        control = _control(core_plan, run_id, payloads)
        control_raw = canonical_json_bytes(control)
        control_sha = sha256_bytes(control_raw)
        attempt_values = {
            "synthetic_run_id": run_id,
            "fixture_control_manifest_sha256": control_sha,
            "logical_time_utc": core_plan["fixture"]["logical_time_utc"],
        }
        if (
            control_sha != EXPECTED_CONTROL_SHA256
            or attempt_values != attempt_authority["values"]
            or opaque_digest(ATTEMPT_DOMAIN, attempt_values)
            != attempt_authority["result"]
        ):
            _stop("R3 control or attempt authority mismatch")

        package = root / "inbox" / run_id / "package"
        control_dir = root / "control" / run_id
        core_builder._prepare_empty_root(root)
        for path in (
            root / "inbox",
            root / "inbox" / run_id,
            package,
            root / "control",
            control_dir,
        ):
            core_builder._mkdir_private(path)
        for name in core_plan["input_package"]["exact_files"]:
            core_builder._write_exclusive(package / name, payloads[name])
        control_path = control_dir / "fixture_control_manifest.json"
        core_builder._write_exclusive(control_path, control_raw)
        for path in (
            package,
            package.parent,
            control_dir,
            control_dir.parent,
            root,
        ):
            fd = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return {
            "synthetic_run_id": run_id,
            "attempt_id": attempt_authority["result"],
            "package_path": str(package),
            "control_manifest_path": str(control_path),
            "fixture_control_manifest_sha256": control_sha,
            "core_plan_sha256": sha256_bytes(core_plan_raw),
            "predecessor_receipt_sha256": sha256_bytes(predecessor_raw),
        }
    finally:
        os.umask(old_umask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=R3_ROOT)
    arguments = parser.parse_args()
    if arguments.root != R3_ROOT:
        _stop("authoritative R3 build requires the exact preregistered root")
    result = build_fixture(root=arguments.root)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
