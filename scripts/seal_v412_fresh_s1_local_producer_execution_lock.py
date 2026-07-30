#!/usr/bin/env python3
"""Build the execution lock for the V4.12 local S1 producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPOSITORY
    / "config/v4_12_fresh_s1_local_producer_authority_plan.json"
)
PROVISIONER_PATH = (
    REPOSITORY / "scripts/provision_v412_fresh_s1_local_producer.py"
)
IMPLEMENTATION_COMMIT = "ad74b4eaeeae1836e9ca08703d442b60454f2682"
TRUSTED_OUTPUT_PARENT = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")


def _load_provisioner():
    spec = importlib.util.spec_from_file_location(
        "s1_lock_provisioner", PROVISIONER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("PROVISIONER_IMPORT_SPEC")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provisioner = _load_provisioner()


class LockSealError(RuntimeError):
    pass


def _stop(reason: str) -> None:
    raise LockSealError(reason)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = provisioner._read_regular(path, label)
        value = provisioner.parse_canonical_object(raw, label)
    except provisioner.ProvisionError as exc:
        _stop(str(exc))
    return value, raw


def _runtime() -> dict[str, Any]:
    try:
        return provisioner._expected_runtime()
    except provisioner.ProvisionError as exc:
        _stop(str(exc))


def _volume_identity(path: Path) -> tuple[int, str]:
    fd = provisioner._open_absolute_dir_anchored(path)
    try:
        return os.fstat(fd).st_dev, provisioner.volume_uuid_for_fd(fd)
    except provisioner.ProvisionError as exc:
        _stop(str(exc))
    finally:
        os.close(fd)


def build_lock(
    plan: Mapping[str, Any],
    plan_raw: bytes,
    *,
    trusted_output_parent: Path,
) -> dict[str, Any]:
    if plan_raw != provisioner.canonical_json(plan):
        _stop("PLAN_NON_CANONICAL")
    contract = plan["authorities"]["contract"]
    contract_path = REPOSITORY / contract["path"]
    contract_raw = provisioner._read_regular(contract_path, "CONTRACT")
    if _sha(contract_raw) != contract["sha256"]:
        _stop("CONTRACT_HASH")
    source_raw = provisioner._read_regular(
        REPOSITORY / plan["future_implementation"]["provisioner"]["path"],
        "PROVISIONER",
    )
    tests_raw = provisioner._read_regular(
        REPOSITORY / plan["future_implementation"]["tests"]["path"],
        "PROVISIONER_TESTS",
    )
    device, volume_uuid = _volume_identity(trusted_output_parent)
    implementation = {
        "git_commit": IMPLEMENTATION_COMMIT,
        "provisioner_path": plan["future_implementation"]["provisioner"][
            "path"
        ],
        "provisioner_sha256": _sha(source_raw),
        "tests_path": plan["future_implementation"]["tests"]["path"],
        "tests_sha256": _sha(tests_raw),
        "plan_path": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_sha256": _sha(plan_raw),
        "contract_path": contract["path"],
        "contract_sha256": contract["sha256"],
    }
    lock = {
        "schema_version": plan["schemas"]["execution_lock"][
            "schema_version"
        ],
        "purpose": "LOCAL_PRODUCER_AUTHORITY_WITHOUT_CRM",
        "plan_path": str(PLAN_PATH.relative_to(REPOSITORY)),
        "plan_sha256": _sha(plan_raw),
        "contract_path": contract["path"],
        "contract_sha256": contract["sha256"],
        "implementation": implementation,
        "runtime": _runtime(),
        "keychain_policy": provisioner._expected_keychain_policy(plan),
        "expected_uid": os.getuid(),
        "volume_device": device,
        "volume_uuid": volume_uuid,
        "output_root": plan["paths"]["root"],
        "logical_time_utc": plan["identity"]["logical_time_utc"],
        "authorization_path": plan["paths"]["authorization"],
    }
    try:
        provisioner._validate_schema(lock, "execution_lock", plan)
    except provisioner.ProvisionError as exc:
        _stop(str(exc))
    serialized = provisioner.canonical_json(lock).decode("utf-8")
    if "UNIMPLEMENTED" in serialized or "<" in serialized or "latest" in serialized:
        _stop("LOCK_SENTINEL")
    return lock


def seal_lock(
    plan_path: Path = PLAN_PATH,
    *,
    output_path: Path | None = None,
    trusted_output_parent: Path = TRUSTED_OUTPUT_PARENT,
) -> dict[str, Any]:
    plan, plan_raw = _read_canonical(plan_path, "PLAN")
    lock = build_lock(
        plan, plan_raw, trusted_output_parent=trusted_output_parent
    )
    destination = (
        REPOSITORY / plan["paths"]["execution_lock"]
        if output_path is None
        else output_path
    )
    parent_fd = provisioner._open_absolute_dir_anchored(destination.parent)
    try:
        provisioner._write_private_at(
            parent_fd, destination.name, provisioner.canonical_json(lock)
        )
    except provisioner.ProvisionError as exc:
        _stop(str(exc))
    finally:
        os.close(parent_fd)
    return lock


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("STOP:ARGS_FORBIDDEN", file=sys.stderr)
        return 64
    try:
        lock = seal_lock()
    except LockSealError as exc:
        print(f"STOP:{exc}", file=sys.stderr)
        return 65
    print(provisioner.canonical_json(lock).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
