#!/usr/bin/env python3
"""Inventory and optionally publish the terminal v13 W3 STOP receipt.

The default ``audit`` command is read-only.  ``publish`` is deliberately a
separate command: it writes one content-addressed receipt beside, never inside,
the historical v13 tree and never changes permissions or bytes in that tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "sireto-retrieval-v13-terminal-stop-receipt-1"
STATUS = "SEALED_TERMINAL_STOP_V13_NO_RETRY"

DEFAULT_TREE_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "retrieval_fold_fused_physical_benchmark_v13_w3"
)
DEFAULT_MARKER = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    ".retrieval_v13_w3_attempt_consumed.json"
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "retrieval_all_human_v13_physical_execution.json"
)
DEFAULT_PUBLICATION_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "retrieval_v13_stop_receipt"
)

EXPECTED_MARKER_SHA256 = (
    "927707c05c82d3f775d887ddf44f7f018f62bce3b5463e6830f0b64878b5b028"
)
EXPECTED_STOP_SHA256 = (
    "4bc15ca54f5bc8aeda211c4aa1c5982c5db94fe0b4bf13a9823a6125a6fba4b9"
)
EXPECTED_CONFIG_SHA256 = (
    "40c75e669289147ecf3b3c89ad37f9ef0f8e289cc092868ec26de10d0e15bf0d"
)
EXPECTED_IMPLEMENTATION_SHA256 = (
    "24841ade82512f1ca8966e8c680b7a0fe2034005c8e2805215fed77c3768a02a"
)
EXPECTED_FILE_COUNT = 47
EXPECTED_TREE_SIZE_BYTES = 155_819_481
EXPECTED_TREE_MODE = "0755"
EXPECTED_INVENTORY_SHA256 = (
    "4b45d3e0793b23a43ae9419cdb669bf9ce96b11828b6e9fc5f51f80f70d1a7e2"
)
EXPECTED_INVENTORY_MERKLE_ROOT = (
    "34c87618e17fafe3347ebf840a059d4100107090c925c14da03fc66cc1effad9"
)
EXPECTED_MARKER_SIZE_BYTES = 466
EXPECTED_MARKER_MODE = "0444"
EXPECTED_STOP_SIZE_BYTES = 4_084
EXPECTED_STOP_MODE = "0644"
EXPECTED_CONFIG_SIZE_BYTES = 26_332
EXPECTED_CONFIG_MODE = "0644"
EXPECTED_JSON_FILE_COUNT = 23
EXPECTED_PROTECTED_MANIFEST_COUNT = 20
EXPECTED_FOLDS = (2, 3, 4)
EXPECTED_EXCEPTION_TYPE = "FoldFusedInvariantError"
EXPECTED_EXCEPTION_MESSAGE = "compact output logical content drift"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class V13StopReceiptError(ValueError):
    """The historical v13 evidence differs from the frozen STOP."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _mode(value: os.stat_result) -> str:
    return f"{stat.S_IMODE(value.st_mode):04o}"


def _stable_regular_file_record(path: Path, *, root: Path) -> dict[str, Any]:
    target = Path(path)
    before = target.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise V13StopReceiptError(f"non-regular or linked evidence: {target}")
    digest = sha256_file(target)
    after = target.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_nlink,
    )
    if identity_before != identity_after:
        raise V13StopReceiptError(f"evidence changed while hashing: {target}")
    try:
        relative = target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise V13StopReceiptError(f"evidence escaped its root: {target}") from error
    return {
        "relative_path": relative,
        "size_bytes": int(after.st_size),
        "sha256": digest,
        "mode": _mode(after),
    }


def inventory_tree(tree_root: Path) -> tuple[dict[str, Any], ...]:
    root = Path(tree_root).resolve()
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise V13StopReceiptError("v13 STOP tree is not a regular directory")
    records: list[dict[str, Any]] = []
    for child in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        observed = child.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise V13StopReceiptError(f"v13 STOP tree contains a symlink: {child}")
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise V13StopReceiptError(
                f"v13 STOP tree contains a non-regular entry: {child}"
            )
        records.append(_stable_regular_file_record(child, root=root))
    if not records:
        raise V13StopReceiptError("v13 STOP tree is empty")
    return tuple(records)


def inventory_merkle_root(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        raise V13StopReceiptError("cannot seal an empty inventory")
    level = [
        hashlib.sha256(b"\x00" + canonical_json_bytes(dict(record))).digest()
        for record in records
    ]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V13StopReceiptError(f"{label} is not readable JSON") from error
    if not isinstance(value, Mapping):
        raise V13StopReceiptError(f"{label} must be a JSON object")
    return value


def _external_record(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if not _valid_sha256(expected_sha256):
        raise V13StopReceiptError("expected authority SHA-256 is malformed")
    target = Path(path).resolve()
    record = _stable_regular_file_record(target, root=target.parent)
    if record["sha256"] != expected_sha256:
        raise V13StopReceiptError(f"authority hash drifted: {target}")
    return {
        "path": str(target),
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "mode": record["mode"],
    }


def _validate_implementation(stop: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = stop.get("implementation_files")
    if not isinstance(raw, list) or len(raw) != 15:
        raise V13StopReceiptError("v13 implementation inventory is incomplete")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value.get("path"), str)
            or not value["path"]
            or Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or not _valid_sha256(value.get("sha256"))
            or type(value.get("size_bytes")) is not int
            or int(value["size_bytes"]) <= 0
        ):
            raise V13StopReceiptError(
                f"v13 implementation record {index} is malformed"
            )
        records.append(
            {
                "path": str(value["path"]),
                "sha256": str(value["sha256"]),
                "size_bytes": int(value["size_bytes"]),
            }
        )
    if (
        sha256_json(records) != EXPECTED_IMPLEMENTATION_SHA256
        or stop.get("v13_implementation_sha256")
        != EXPECTED_IMPLEMENTATION_SHA256
        or stop.get("post_error_current_implementation_sha256")
        != EXPECTED_IMPLEMENTATION_SHA256
    ):
        raise V13StopReceiptError("v13 implementation aggregate drifted")
    return tuple(records)


def _audit_tree_safety(
    *, tree_root: Path, inventory: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    root = Path(tree_root).resolve()
    json_count = 0
    protected_count = 0
    for record in inventory:
        relative = str(record["relative_path"])
        if not relative.endswith(".json"):
            continue
        json_count += 1
        value = _read_json(root / relative, label=f"v13 JSON {relative}")
        protected = any(
            name in value
            for name in (
                "fold0_opened",
                "fold1_opened",
                "positive_injection",
                "execution_authorization_published",
            )
        )
        if protected:
            protected_count += 1
        if value.get("fold0_opened", False) is not False:
            raise V13StopReceiptError(f"fold0 was opened in {relative}")
        if value.get("fold1_opened", False) is not False:
            raise V13StopReceiptError(f"fold1 was opened in {relative}")
        if value.get("positive_injection", False) is not False:
            raise V13StopReceiptError(f"positive injection detected in {relative}")
        if value.get("execution_authorization_published", False) is not False:
            raise V13StopReceiptError(
                f"execution authorization published in {relative}"
            )
        folds = value.get("folds")
        if isinstance(folds, list) and any(int(fold) in {0, 1} for fold in folds):
            raise V13StopReceiptError(f"closed fold present in {relative}")
    authorization_paths = [
        str(record["relative_path"])
        for record in inventory
        if Path(str(record["relative_path"])).name
        == "execution_authorization.json"
    ]
    if authorization_paths:
        raise V13StopReceiptError("v13 tree contains an execution authorization")
    return {
        "json_file_count": json_count,
        "protected_manifest_count": protected_count,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "execution_authorization_present": False,
        "execution_authorization_published": False,
    }


def build_receipt(
    *,
    tree_root: Path,
    marker_path: Path,
    config_path: Path,
    expected_tree_file_count: int = EXPECTED_FILE_COUNT,
    expected_tree_size_bytes: int = EXPECTED_TREE_SIZE_BYTES,
    expected_tree_mode: str = EXPECTED_TREE_MODE,
    expected_inventory_sha256: str = EXPECTED_INVENTORY_SHA256,
    expected_inventory_merkle_root: str = EXPECTED_INVENTORY_MERKLE_ROOT,
    expected_marker_sha256: str = EXPECTED_MARKER_SHA256,
    expected_marker_size_bytes: int = EXPECTED_MARKER_SIZE_BYTES,
    expected_marker_mode: str = EXPECTED_MARKER_MODE,
    expected_stop_sha256: str = EXPECTED_STOP_SHA256,
    expected_stop_size_bytes: int = EXPECTED_STOP_SIZE_BYTES,
    expected_stop_mode: str = EXPECTED_STOP_MODE,
    expected_config_sha256: str = EXPECTED_CONFIG_SHA256,
    expected_config_size_bytes: int = EXPECTED_CONFIG_SIZE_BYTES,
    expected_config_mode: str = EXPECTED_CONFIG_MODE,
    expected_json_file_count: int = EXPECTED_JSON_FILE_COUNT,
    expected_protected_manifest_count: int = EXPECTED_PROTECTED_MANIFEST_COUNT,
    receipt_builder_path: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic receipt entirely in memory, without writing."""

    root = Path(tree_root).resolve()
    stop_path = root / "manifest.json"
    inventory = inventory_tree(root)
    total_bytes = sum(int(value["size_bytes"]) for value in inventory)
    inventory_sha256 = sha256_json(list(inventory))
    merkle_root = inventory_merkle_root(inventory)
    root_stat = root.lstat()
    if (
        len(inventory) != expected_tree_file_count
        or total_bytes != expected_tree_size_bytes
        or _mode(root_stat) != expected_tree_mode
        or inventory_sha256 != expected_inventory_sha256
        or merkle_root != expected_inventory_merkle_root
    ):
        raise V13StopReceiptError(
            "v13 STOP tree inventory, Merkle, size, count, or mode drifted"
        )
    marker_record = _external_record(
        marker_path, expected_sha256=expected_marker_sha256
    )
    stop_record = _external_record(
        stop_path, expected_sha256=expected_stop_sha256
    )
    config_record = _external_record(
        config_path, expected_sha256=expected_config_sha256
    )
    if (
        marker_record["size_bytes"] != expected_marker_size_bytes
        or marker_record["mode"] != expected_marker_mode
        or stop_record["size_bytes"] != expected_stop_size_bytes
        or stop_record["mode"] != expected_stop_mode
        or config_record["size_bytes"] != expected_config_size_bytes
        or config_record["mode"] != expected_config_mode
    ):
        raise V13StopReceiptError("v13 authority size or mode drifted")
    marker = _read_json(Path(marker_record["path"]), label="v13 attempt marker")
    stop = _read_json(stop_path, label="v13 terminal STOP")
    if (
        marker.get("schema_version")
        != "sireto-retrieval-v13-one-shot-attempt-1"
        or marker.get("status")
        != "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK"
        or marker.get("config_sha256") != expected_config_sha256
        or marker.get("canonical_output_dir") != str(root)
        or marker.get("marker_path") != str(Path(marker_path).resolve())
        or marker.get("retry_allowed") is not False
    ):
        raise V13StopReceiptError("v13 one-shot marker contract drifted")
    exception = stop.get("exception")
    stop_config = stop.get("config")
    stop_marker = stop.get("attempt_marker")
    if (
        stop.get("schema_version")
        != "sireto-retrieval-v13-execution-exception-1"
        or stop.get("status") != "STOP_V13_EXECUTION_EXCEPTION"
        or stop.get("phase") != "MEASURED_W3_CALLBACK_OR_POSTCHECK"
        or stop.get("folds") != list(EXPECTED_FOLDS)
        or stop.get("fold0_opened") is not False
        or stop.get("fold1_opened") is not False
        or stop.get("positive_injection") is not False
        or stop.get("execution_authorization_published") is not False
        or stop.get("automatic_retry_allowed") is not False
        or not isinstance(stop_config, Mapping)
        or stop_config.get("path") != config_record["path"]
        or stop_config.get("sha256") != expected_config_sha256
        or not isinstance(stop_marker, Mapping)
        or stop_marker.get("path") != marker_record["path"]
        or stop_marker.get("sha256") != expected_marker_sha256
        or not isinstance(exception, Mapping)
        or exception.get("type") != EXPECTED_EXCEPTION_TYPE
        or exception.get("message_sanitized") != EXPECTED_EXCEPTION_MESSAGE
    ):
        raise V13StopReceiptError("v13 terminal STOP contract drifted")
    implementation = _validate_implementation(stop)
    safety = _audit_tree_safety(tree_root=root, inventory=inventory)
    if (
        safety["json_file_count"] != expected_json_file_count
        or safety["protected_manifest_count"]
        != expected_protected_manifest_count
    ):
        raise V13StopReceiptError("v13 JSON safety inventory drifted")
    builder_record = (
        _external_record(
            Path(receipt_builder_path).resolve(),
            expected_sha256=sha256_file(Path(receipt_builder_path).resolve()),
        )
        if receipt_builder_path is not None
        else None
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "terminal_decision": "STOP_V13_EXECUTION_EXCEPTION",
        "retry_allowed": False,
        "v13_outputs_reusable_for_another_attempt": False,
        "v13_execution_authorization_valid": False,
        "source_tree": {
            "path": str(root),
            "mode": _mode(root_stat),
            "file_count": len(inventory),
            "total_size_bytes": total_bytes,
            "inventory_sha256": inventory_sha256,
            "inventory_merkle_root": merkle_root,
        },
        "authorities": {
            "attempt_marker": marker_record,
            "terminal_stop_manifest": stop_record,
            "physical_config": config_record,
            "v13_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
            "v13_implementation_files": list(implementation),
            "v13_runtime_commit": "41b8eed9ca8683a098a6757b54fdc5411bf69ae9",
            "v13_handover_commit": "91a18cf2ec9e937779b96f120ea08bf460765456",
        },
        "safety": safety,
        "failure": {
            "phase": "MEASURED_W3_CALLBACK_OR_POSTCHECK",
            "exception_type": EXPECTED_EXCEPTION_TYPE,
            "message_sanitized": EXPECTED_EXCEPTION_MESSAGE,
        },
        "receipt_builder": builder_record,
        "inventory": list(inventory),
    }
    return {**body, "receipt_id": sha256_json(body)}


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate a receipt independently before publication."""

    if set(receipt) != {
        "schema_version",
        "status",
        "terminal_decision",
        "retry_allowed",
        "v13_outputs_reusable_for_another_attempt",
        "v13_execution_authorization_valid",
        "source_tree",
        "authorities",
        "safety",
        "failure",
        "receipt_builder",
        "inventory",
        "receipt_id",
    }:
        raise V13StopReceiptError("receipt field set drifted")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt_id = str(receipt.get("receipt_id") or "")
    if not _valid_sha256(receipt_id) or receipt_id != sha256_json(body):
        raise V13StopReceiptError("receipt content address drifted")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != STATUS
        or receipt.get("terminal_decision") != "STOP_V13_EXECUTION_EXCEPTION"
        or receipt.get("retry_allowed") is not False
        or receipt.get("v13_outputs_reusable_for_another_attempt") is not False
        or receipt.get("v13_execution_authorization_valid") is not False
    ):
        raise V13StopReceiptError("receipt terminal decision drifted")
    inventory = receipt.get("inventory")
    source_tree = receipt.get("source_tree")
    safety = receipt.get("safety")
    if (
        not isinstance(inventory, list)
        or not inventory
        or not isinstance(source_tree, Mapping)
        or not isinstance(safety, Mapping)
    ):
        raise V13StopReceiptError("receipt inventory or safety block is malformed")
    expected_keys = {"relative_path", "size_bytes", "sha256", "mode"}
    relative_paths: list[str] = []
    for index, record in enumerate(inventory):
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_keys
            or not isinstance(record.get("relative_path"), str)
            or not record["relative_path"]
            or Path(record["relative_path"]).is_absolute()
            or ".." in Path(record["relative_path"]).parts
            or type(record.get("size_bytes")) is not int
            or int(record["size_bytes"]) < 0
            or not _valid_sha256(record.get("sha256"))
            or not isinstance(record.get("mode"), str)
            or re.fullmatch(r"[0-7]{4}", str(record["mode"])) is None
        ):
            raise V13StopReceiptError(f"receipt inventory record {index} is malformed")
        relative_paths.append(str(record["relative_path"]))
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(
        set(relative_paths)
    ):
        raise V13StopReceiptError("receipt inventory order or uniqueness drifted")
    total_bytes = sum(int(record["size_bytes"]) for record in inventory)
    if (
        source_tree.get("file_count") != len(inventory)
        or source_tree.get("total_size_bytes") != total_bytes
        or source_tree.get("inventory_sha256") != sha256_json(inventory)
        or source_tree.get("inventory_merkle_root")
        != inventory_merkle_root(inventory)
        or safety.get("fold0_opened") is not False
        or safety.get("fold1_opened") is not False
        or safety.get("positive_injection") is not False
        or safety.get("execution_authorization_present") is not False
        or safety.get("execution_authorization_published") is not False
    ):
        raise V13StopReceiptError("receipt inventory or safety seal drifted")


def publish_receipt(receipt: Mapping[str, Any], *, output_root: Path) -> Path:
    """Atomically publish one immutable receipt outside the historical tree."""

    validate_receipt(receipt)
    receipt_id = str(receipt["receipt_id"])
    source_root = Path(str(receipt["source_tree"]["path"])).resolve()
    root = Path(output_root).resolve()
    if root == source_root or source_root in root.parents or root in source_root.parents:
        raise V13StopReceiptError(
            "receipt publication root must be separate from the v13 tree"
        )
    payload = canonical_json_bytes(dict(receipt)) + b"\n"
    target = root / receipt_id[:16]
    manifest_path = target / "manifest.json"
    if target.exists():
        if (
            not target.is_dir()
            or target.is_symlink()
            or {child.name for child in target.iterdir()} != {"manifest.json"}
            or not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.read_bytes() != payload
            or _mode(manifest_path.lstat()) != "0444"
        ):
            raise V13StopReceiptError("receipt content-address collision")
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v13-stop-receipt-", dir=root))
    promoted = False
    try:
        temporary_manifest = temporary / "manifest.json"
        descriptor = os.open(
            temporary_manifest,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise V13StopReceiptError(
                        "short write while sealing receipt"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary_manifest, 0o444)
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(temporary, 0o555)
        os.rename(temporary, target)
        promoted = True
        parent_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return manifest_path
    finally:
        if not promoted and temporary.exists():
            os.chmod(temporary, 0o755)
            shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or publish the immutable terminal v13 W3 STOP receipt."
    )
    parser.add_argument("command", choices=("audit", "publish"))
    parser.add_argument("--tree-root", type=Path, default=DEFAULT_TREE_ROOT)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_PUBLICATION_ROOT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_receipt(
            tree_root=args.tree_root,
            marker_path=args.marker,
            config_path=args.config,
            receipt_builder_path=Path(__file__).resolve(),
        )
        if args.command == "audit":
            print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        path = publish_receipt(receipt, output_root=args.output_root)
        result = {
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
            "receipt_id": receipt["receipt_id"],
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except V13StopReceiptError as error:
        print(f"STOP_V13_TOMBSTONE: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
