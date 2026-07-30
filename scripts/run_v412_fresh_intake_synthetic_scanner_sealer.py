#!/usr/bin/env python3
"""Run the V4.12 S0 synthetic scanner/sealer, fail closed.

No real CRM path is accepted.  The production CLI always performs the
single-process 60-second stability observation.  Tests may inject a shorter
wait only by calling ``run_scanner(..., _test_mode=True)`` directly.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scripts.build_v412_fresh_intake_synthetic_fixture import (
        FORBIDDEN_COMPONENTS,
        PLAN_DEFAULT,
        canonical_json_bytes,
        opaque_digest,
        sha256_bytes,
    )
except ModuleNotFoundError:
    from build_v412_fresh_intake_synthetic_fixture import (
        FORBIDDEN_COMPONENTS,
        PLAN_DEFAULT,
        canonical_json_bytes,
        opaque_digest,
        sha256_bytes,
    )


ID_RE = re.compile(r"^[a-p]{64}$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
CSV_HEADER = [
    "source_batch_id",
    "source_record_id",
    "source_system",
    "portfolio_id",
    "crm_name_raw",
    "crm_address_raw",
    "crm_postcode_raw",
    "crm_city_raw",
    "crm_insee_raw",
]
class ScannerStop(RuntimeError):
    """Fail-closed scanner error."""


def _stop(message: str) -> None:
    if not message.upper().startswith("STOP"):
        message = f"STOP {message}"
    raise ScannerStop(message)


def _audit_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _is_strict_rfc3339_utc_seconds(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value


def _audit_timestamp(value: Any) -> str:
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            _stop("audit clock returned a naive datetime")
        return (
            value.astimezone(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if _is_strict_rfc3339_utc_seconds(value):
        return value
    _stop("audit clock must return RFC3339 UTC seconds or aware datetime")


def _decode_json_canonical(
    raw: bytes, label: str
) -> tuple[dict[str, Any], bytes]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _stop(f"invalid JSON {label}: {exc}")
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        _stop(f"non-canonical JSON object: {label}")
    return value, raw


def _load_json_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    return _decode_json_canonical(path.read_bytes(), str(path))


def _load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    plan, raw = _load_json_canonical(path)
    contract_path = Path(plan["contract"]["path"])
    if sha256_bytes(contract_path.read_bytes()) != plan["contract"]["sha256"]:
        _stop("contract pin mismatch")
    if sha256_bytes(
        canonical_json_bytes(plan["fixture"], final_lf=False)
    ) != plan["control_manifest"]["fixture_spec_sha256"]:
        _stop("fixture spec pin mismatch")
    return plan, raw


def _validate_root(root: Path) -> Path:
    if not root.is_absolute():
        _stop("synthetic root must be absolute")
    normalized = Path(os.path.abspath(os.fspath(root)))
    if any(part in FORBIDDEN_COMPONENTS for part in normalized.parts):
        _stop("synthetic root contains a forbidden real-data component")
    repository = Path(__file__).resolve().parents[1]
    try:
        normalized.relative_to(repository)
    except ValueError:
        pass
    else:
        _stop("synthetic root must be outside the repository")
    return normalized


class _RootFD:
    """Deny-by-construction filesystem authority anchored at one root FD."""

    def __init__(self, root: Path):
        self.path = root
        if not root.is_absolute():
            _stop("root authority path must be absolute")
        components = tuple(part for part in root.parts if part != "/")
        current = os.open(
            "/",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        self._absolute_chain: list[tuple[int, int]] = []
        try:
            for component in components:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
                current_info = os.fstat(current)
                if not stat.S_ISDIR(current_info.st_mode):
                    _stop("non-directory in absolute root chain")
                self._absolute_chain.append(
                    (current_info.st_dev, current_info.st_ino)
                )
            self.fd = current
        except Exception:
            os.close(current)
            raise
        info = os.fstat(self.fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            os.close(self.fd)
            _stop("unsafe synthetic root")
        self.device = info.st_dev

    def require_beneath(self, prefix: Path) -> None:
        if not prefix.is_absolute():
            _stop("test prefix must be absolute")
        prefix_components = tuple(
            part for part in prefix.parts if part != "/"
        )
        root_components = tuple(
            part for part in self.path.parts if part != "/"
        )
        if (
            len(root_components) <= len(prefix_components)
            or root_components[: len(prefix_components)] != prefix_components
        ):
            _stop("root is not strictly below the authorised test prefix")
        current = os.open(
            "/",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            for component in prefix_components:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
            info = os.fstat(current)
            if (
                self._absolute_chain[len(prefix_components) - 1]
                != (info.st_dev, info.st_ino)
            ):
                _stop("authorised test prefix FD identity mismatch")
        finally:
            os.close(current)

    def close(self) -> None:
        os.close(self.fd)

    @staticmethod
    def _parts(relative: str | Sequence[str]) -> tuple[str, ...]:
        parts = (
            tuple(relative)
            if not isinstance(relative, str)
            else tuple(part for part in relative.split("/") if part)
        )
        if not parts or any(
            not part or part in {".", ".."} or "/" in part for part in parts
        ):
            _stop("unsafe root-relative path")
        return parts

    def _validate_dir_fd(self, fd: int) -> None:
        info = os.fstat(fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_dev != self.device
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _stop("directory escaped root authority")

    def open_dir(
        self,
        relative: str | Sequence[str],
        *,
        create: bool = False,
    ) -> int:
        current = os.dup(self.fd)
        try:
            for part in self._parts(relative):
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
                self._validate_dir_fd(current)
            return current
        except Exception:
            os.close(current)
            raise

    def list(self, relative: str | Sequence[str]) -> list[str]:
        fd = self.open_dir(relative)
        try:
            return sorted(os.listdir(fd))
        finally:
            os.close(fd)

    def exists(
        self, relative: str | Sequence[str], *, directory: bool
    ) -> bool:
        parts = self._parts(relative)
        try:
            parent = (
                self.open_dir(parts[:-1])
                if len(parts) > 1
                else os.dup(self.fd)
            )
        except FileNotFoundError:
            return False
        try:
            try:
                info = os.stat(
                    parts[-1], dir_fd=parent, follow_symlinks=False
                )
            except FileNotFoundError:
                return False
            expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
            if (
                not expected
                or info.st_uid != os.getuid()
                or info.st_dev != self.device
                or (not directory and info.st_nlink != 1)
                or stat.S_IMODE(info.st_mode)
                != (0o700 if directory else 0o600)
            ):
                _stop("unsafe object under root authority")
            return True
        finally:
            os.close(parent)

    def read_file(self, relative: str | Sequence[str]) -> bytes:
        parts = self._parts(relative)
        parent = self.open_dir(parts[:-1]) if len(parts) > 1 else os.dup(self.fd)
        try:
            fd = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_dev != self.device
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    _stop("unsafe file under root authority")
                raw = _read_fd(fd)
                if len(raw) != info.st_size:
                    _stop("short anchored read")
                return raw
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def write_exclusive(
        self, relative: str | Sequence[str], payload: bytes
    ) -> None:
        parts = self._parts(relative)
        parent = self.open_dir(parts[:-1], create=True)
        try:
            fd = os.open(
                parts[-1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent,
            )
            try:
                view = memoryview(payload)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        _stop("short anchored write")
                    view = view[count:]
                os.fchmod(fd, 0o600)
                _sync_fd(fd, full_required=True)
            finally:
                os.close(fd)
            _sync_fd(parent, full_required=False)
        finally:
            os.close(parent)

    def mkdir_exclusive(self, relative: str | Sequence[str]) -> None:
        parts = self._parts(relative)
        parent = self.open_dir(parts[:-1], create=True)
        try:
            os.mkdir(parts[-1], 0o700, dir_fd=parent)
            _sync_fd(parent, full_required=False)
        finally:
            os.close(parent)

    def rename_exclusive(
        self,
        source: str | Sequence[str],
        destination: str | Sequence[str],
    ) -> None:
        source_parts = self._parts(source)
        destination_parts = self._parts(destination)
        source_parent = self.open_dir(source_parts[:-1])
        destination_parent = self.open_dir(
            destination_parts[:-1], create=True
        )
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            function = getattr(libc, "renameatx_np", None)
            if function is None:
                _stop("renameatx_np unavailable")
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            result = function(
                source_parent,
                os.fsencode(source_parts[-1]),
                destination_parent,
                os.fsencode(destination_parts[-1]),
                0x00000004,
            )
            if result:
                _stop(
                    "exclusive promotion failed: "
                    + os.strerror(ctypes.get_errno())
                )
            _sync_fd(source_parent, full_required=False)
            _sync_fd(destination_parent, full_required=False)
        finally:
            os.close(source_parent)
            os.close(destination_parent)


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _sync_fd(fd: int, *, full_required: bool) -> None:
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is None:
        if full_required and os.uname().sysname == "Darwin":
            _stop("F_FULLFSYNC unavailable")
        return
    try:
        fcntl.fcntl(fd, full)
    except OSError:
        if full_required:
            _stop("F_FULLFSYNC failed")


def _open_stable_payloads(
    authority: _RootFD,
    package: str,
    names: Sequence[str],
    wait_seconds: float,
    sleep_fn: Callable[[float], Any],
) -> dict[str, bytes]:
    if set(authority.list(package)) != set(names):
        _stop("input package exact-tree mismatch")
    directory_fd = authority.open_dir(package)
    opened: dict[str, int] = {}
    try:
        first: dict[str, tuple[os.stat_result, bytes]] = {}
        for name in names:
            fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            opened[name] = fd
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or info.st_dev != authority.device
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                _stop(f"unsafe input payload: {name}")
            data = _read_fd(fd)
            if len(data) != info.st_size:
                _stop(f"input size changed during first read: {name}")
            if len(data) > 4 * 1024 * 1024:
                _stop(f"input payload exceeds 4 MiB: {name}")
            first[name] = (info, data)
        devices = {item[0].st_dev for item in first.values()}
        if len(devices) != 1:
            _stop("input payloads span devices")
        started = time.monotonic()
        sleep_fn(wait_seconds)
        if time.monotonic() - started + 1e-6 < wait_seconds:
            _stop("stability interval was not observed")
        result: dict[str, bytes] = {}
        for name in names:
            fd = opened[name]
            before, prior = first[name]
            after = os.fstat(fd)
            path_info = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
            if identity_before != identity_after or (
                path_info.st_dev,
                path_info.st_ino,
            ) != (after.st_dev, after.st_ino):
                _stop(f"input mutation or substitution detected: {name}")
            current = _read_fd(fd)
            if current != prior or len(current) != after.st_size:
                _stop(f"input bytes changed during stability: {name}")
            result[name] = current
        return result
    finally:
        for fd in opened.values():
            os.close(fd)
        os.close(directory_fd)


def _payload_manifest(
    *,
    plan: Mapping[str, Any],
    package_kind: str,
    run_id: str,
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    fixture = plan["fixture"]
    records = [
        {
            "relative_path": name,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in payloads.items()
    ]
    tree_hash = sha256_bytes(canonical_json_bytes(records, final_lf=False))
    manifest = {
        "schema_version": "sireto-v4.12-fresh-synthetic-payload-manifest-1",
        "package_kind": package_kind,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "source_batch_id": fixture["common_provenance"]["source_batch_id"],
        "logical_time_utc": fixture["logical_time_utc"],
        "ordered_payload_records": records,
        "payload_count": len(records),
        "payload_tree_sha256": tree_hash,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    seal = {
        "schema_version": "sireto-v4.12-fresh-synthetic-seal-1",
        "package_kind": package_kind,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "source_batch_id": fixture["common_provenance"]["source_batch_id"],
        "logical_time_utc": fixture["logical_time_utc"],
        "payload_manifest_size_bytes": len(manifest_bytes),
        "payload_manifest_sha256": sha256_bytes(manifest_bytes),
        "payload_tree_sha256": tree_hash,
    }
    return manifest, manifest_bytes, seal, canonical_json_bytes(seal)


def _seal_tree(
    *,
    plan: Mapping[str, Any],
    authority: _RootFD,
    run_id: str,
    destination: str,
    package_kind: str,
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    temp_parent = f"tmp/{run_id}"
    temp_fd = authority.open_dir(temp_parent, create=True)
    os.close(temp_fd)
    prefix = (
        package_kind.lower()
        + "-"
        + sha256_bytes(
            canonical_json_bytes(list(payloads), final_lf=False)
        )[:16]
        + "-"
    )
    staging = f"{temp_parent}/{prefix}{os.urandom(8).hex()}"
    authority.mkdir_exclusive(staging)
    _manifest, manifest_bytes, _seal, seal_bytes = _payload_manifest(
        plan=plan,
        package_kind=package_kind,
        run_id=run_id,
        payloads=payloads,
    )
    for name, payload in payloads.items():
        authority.write_exclusive(f"{staging}/{name}", payload)
    authority.write_exclusive(
        f"{staging}/payload_manifest.json", manifest_bytes
    )
    authority.write_exclusive(f"{staging}/seal.json", seal_bytes)
    _validate_tree(
        authority,
        staging,
        package_kind=package_kind,
        expected_payload_names=list(payloads),
        plan=plan,
        run_id=run_id,
    )
    authority.rename_exclusive(staging, destination)
    return {
        "payload_manifest_sha256": sha256_bytes(manifest_bytes),
        "seal_sha256": sha256_bytes(seal_bytes),
    }


def _validate_tree(
    authority: _RootFD,
    tree: str,
    *,
    package_kind: str,
    expected_payload_names: Sequence[str],
    plan: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    expected = set(expected_payload_names) | {"payload_manifest.json", "seal.json"}
    if set(authority.list(tree)) != expected:
        _stop(f"{package_kind} exact-tree mismatch")
    payloads: dict[str, bytes] = {}
    for name in expected_payload_names:
        payloads[name] = authority.read_file(f"{tree}/{name}")
    manifest, manifest_bytes = _decode_json_canonical(
        authority.read_file(f"{tree}/payload_manifest.json"),
        f"{tree}/payload_manifest.json",
    )
    seal, seal_bytes = _decode_json_canonical(
        authority.read_file(f"{tree}/seal.json"), f"{tree}/seal.json"
    )
    expected_manifest, expected_manifest_bytes, expected_seal, expected_seal_bytes = (
        _payload_manifest(
            plan=plan,
            package_kind=package_kind,
            run_id=run_id,
            payloads=payloads,
        )
    )
    if (
        manifest != expected_manifest
        or manifest_bytes != expected_manifest_bytes
        or seal != expected_seal
        or seal_bytes != expected_seal_bytes
    ):
        _stop(f"{package_kind} manifest or seal mismatch")
    return {
        "payloads": payloads,
        "payload_manifest_sha256": sha256_bytes(manifest_bytes),
        "seal_sha256": sha256_bytes(seal_bytes),
    }


def _validate_source_manifests(
    plan: Mapping[str, Any],
    control: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    hash_fields = {
        "collection_source_manifest.json": (
            "collection_source_manifest_sha256"
        ),
        "source_manifest.json": "source_manifest_sha256",
        "crm_safe.csv": "crm_safe_csv_sha256",
        "evidence_source_manifest.json": (
            "evidence_source_manifest_sha256"
        ),
        "evidence_source.parquet": "evidence_source_parquet_sha256",
    }
    for name, field in hash_fields.items():
        if sha256_bytes(payloads[name]) != control[field]:
            _stop(f"control hash mismatch: {name}")
    decoded: dict[str, dict[str, Any]] = {}
    for name in (
        "collection_source_manifest.json",
        "source_manifest.json",
        "evidence_source_manifest.json",
    ):
        try:
            value = json.loads(payloads[name])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _stop(f"invalid source manifest {name}: {exc}")
        if canonical_json_bytes(value) != payloads[name]:
            _stop(f"non-canonical source manifest: {name}")
        decoded[name] = value
    collection = decoded["collection_source_manifest.json"]
    source = decoded["source_manifest.json"]
    evidence = decoded["evidence_source_manifest.json"]
    manifests_plan = plan["manifests"]
    for label, value in (
        ("collection", collection),
        ("source", source),
        ("evidence", evidence),
    ):
        if list(value) != sorted(value):
            _stop(f"{label} manifest key order is not canonical")
        if set(value) != set(manifests_plan[f"{label}_exact_fields"]):
            _stop(f"{label} manifest fields mismatch")
        if any(item is None for item in value.values()):
            _stop(f"{label} manifest contains null")
        if value["synthetic_run_id"] != run_id:
            _stop(f"{label} manifest run mismatch")
        _validate_field_specs(
            value, manifests_plan[f"{label}_field_types"], label
        )
    fixture = plan["fixture"]
    provenance = fixture["common_provenance"]
    if (
        source["source_sha256"] != sha256_bytes(payloads["crm_safe.csv"])
        or source["source_size_bytes"] != len(payloads["crm_safe.csv"])
        or evidence["evidence_sha256"]
        != sha256_bytes(payloads["evidence_source.parquet"])
        or evidence["evidence_size_bytes"]
        != len(payloads["evidence_source.parquet"])
        or source["source_record_id_semantics"]
        != "UNIQUE_WITHIN_BATCH_OPAQUE_SOURCE_IDENTIFIER"
        or collection["collection_id"] != fixture["collection_id"]
        or source["source_batch_id"] != provenance["source_batch_id"]
        or evidence["source_batch_id"] != provenance["source_batch_id"]
    ):
        _stop("source manifest semantic mismatch")
    schema = pa.schema(
        [
            pa.field(name, _arrow_type(type_name), nullable=nullable)
            for name, type_name, nullable in plan["evidence_parquet"][
                "exact_schema"
            ]
        ]
    )
    parquet = pq.ParquetFile(io.BytesIO(payloads["evidence_source.parquet"]))
    if (
        parquet.schema_arrow != schema
        or parquet.schema_arrow.metadata not in (None, {})
        or parquet.metadata.num_rows != 0
        or parquet.metadata.num_row_groups != 1
    ):
        _stop("evidence Parquet schema or physical shape mismatch")
    file_metadata = parquet.metadata.metadata or {}
    allowed_file_metadata = (
        {b"ARROW:schema"} if plan["parquet_writer"]["store_schema"] else set()
    )
    if set(file_metadata) - allowed_file_metadata:
        _stop("evidence Parquet application metadata forbidden")
    return collection, source, evidence


def _validate_field_specs(
    value: Mapping[str, Any],
    specifications: Mapping[str, str],
    label: str,
) -> None:
    if set(value) != set(specifications):
        _stop(f"{label} typed field coverage mismatch")
    for field, specification in specifications.items():
        item = value[field]
        if specification.startswith("const_uint64:"):
            expected = int(specification.split(":", 1)[1])
            valid = type(item) is int and item >= 0 and item == expected
        elif specification.startswith("const_bool:"):
            expected = specification.endswith(":true")
            valid = type(item) is bool and item is expected
        elif specification.startswith("const_array:"):
            encoded = specification.split(":", 1)[1]
            if encoded == "[]":
                expected_array: list[str] = []
            else:
                expected_array = encoded[1:-1].split(",")
            valid = type(item) is list and item == expected_array
        elif specification.startswith("const:"):
            valid = type(item) is str and item == specification.split(":", 1)[1]
        elif specification == "a_p_64":
            valid = type(item) is str and ID_RE.fullmatch(item) is not None
        elif specification == "hex64":
            valid = type(item) is str and HEX_RE.fullmatch(item) is not None
        elif specification == "uint64":
            valid = type(item) is int and 0 <= item < 2**64
        elif specification == "string_nonempty":
            valid = type(item) is str and bool(item)
        elif specification.startswith("enum:"):
            valid = type(item) is str and item in specification[5:].split("|")
        elif specification == "rfc3339_utc_seconds":
            valid = type(item) is str and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", item
            )
        else:
            _stop(f"unsupported typed specification: {specification}")
        if not valid:
            _stop(f"{label}.{field} violates {specification}")


def _arrow_type(name: str) -> pa.DataType:
    mapping = {
        "string": pa.string(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
    }
    try:
        return mapping[name]
    except KeyError:
        _stop(f"unsupported Arrow type: {name}")


def _batch_parse(
    payload: bytes,
) -> tuple[str | None, list[list[str]]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "CSV_ENCODING_INVALID", []
    if payload.startswith(b"\xef\xbb\xbf"):
        return "CSV_BOM_FORBIDDEN", []
    if b"\x00" in payload:
        return "CSV_NUL_FORBIDDEN", []
    has_crlf = "\r\n" in text
    residual = text.replace("\r\n", "")
    if "\r" in residual or (has_crlf and "\n" in residual):
        return "CSV_MIXED_LINE_ENDINGS", []
    if any(
        len(line.encode("utf-8")) > 64 * 1024
        for line in text.splitlines()
    ):
        return "CSV_ROW_SHAPE_DRIFT", []
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error:
        return "CSV_ROW_SHAPE_DRIFT", []
    if not rows or rows[0] != CSV_HEADER:
        return "CSV_HEADER_DRIFT", []
    if any(len(row) != len(CSV_HEADER) for row in rows[1:]):
        return "CSV_ROW_SHAPE_DRIFT", []
    if any(
        len(cell.encode("utf-8")) > 16 * 1024
        for row in rows
        for cell in row
    ):
        return "CSV_ROW_SHAPE_DRIFT", []
    return None, rows[1:]


def _has_decimal_leak(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value)
    index = 0
    while index < len(normalized):
        if not normalized[index].isdecimal():
            index += 1
            continue
        end = index
        while end < len(normalized) and normalized[end].isdecimal():
            end += 1
        if end - index in {9, 14}:
            return True
        index = end
    return False


def _fresh_opaque_spec(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = Path(plan["references"]["fresh_plan"]["path"])
    raw = reference.read_bytes()
    if sha256_bytes(raw) != plan["references"]["fresh_plan"]["sha256"]:
        _stop("fresh plan pin mismatch")
    fresh = json.loads(raw)
    return fresh["opaque_ids"]


def _implementation_bundle_sha256() -> str:
    producer = Path(__file__).with_name(
        "build_v412_fresh_intake_synthetic_fixture.py"
    ).read_bytes()
    scanner = Path(__file__).read_bytes()
    return sha256_bytes(
        canonical_json_bytes(
            {
                "producer_sha256": sha256_bytes(producer),
                "scanner_sha256": sha256_bytes(scanner),
            },
            final_lf=False,
        )
    )


def _tests_sha256() -> str:
    test_path = Path(__file__).resolve().parents[1] / "tests" / (
        "test_v412_fresh_intake_synthetic_scanner_sealer.py"
    )
    if not test_path.is_file():
        _stop("S0 test source is unavailable")
    return sha256_bytes(test_path.read_bytes())


def _raw_row_hash(row: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(row), final_lf=False))


def _scan_rows(
    plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    rows: Sequence[Sequence[str]],
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    fixture = plan["fixture"]
    provenance = fixture["common_provenance"]
    opaque = _fresh_opaque_spec(plan)
    counts = Counter(row[1] for row in rows)
    reason_order = plan["quarantine"]["row_reason_order"]
    reason_counts = {
        reason: 0 for reason in plan["scan"]["typed_json"]["reason_counts"]["keys"]
    }
    safe: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    batch_payload = {
        "collection_id": fixture["collection_id"],
        "export_snapshot_id": fixture["export_snapshot_id"],
        "source_batch_id": provenance["source_batch_id"],
    }
    batch_id = opaque_digest(opaque["domains"]["batch"], batch_payload)
    stratum_payload = {
        "collection_id": fixture["collection_id"],
        "source_system": provenance["source_system"],
        "portfolio_id": provenance["portfolio_id"],
    }
    stratum_id = opaque_digest(opaque["domains"]["stratum"], stratum_payload)
    for number, values in enumerate(rows, start=1):
        row = dict(zip(CSV_HEADER, values, strict=True))
        raw_hash = _raw_row_hash(values)
        query_payload = {
            "collection_id": fixture["collection_id"],
            "source_batch_id": row["source_batch_id"],
            "source_record_id": row["source_record_id"],
            "source_sha256": source_manifest["source_sha256"],
            "raw_row_sha256": raw_hash,
        }
        query_id = opaque_digest(opaque["domains"]["query"], query_payload)
        locator = sha256_bytes(
            plan["scan"]["locator"]["domain"].encode("utf-8")
            + canonical_json_bytes(
                {
                    "synthetic_run_id": run_id,
                    "source_batch_id": row["source_batch_id"],
                    "source_row_number": number,
                    "raw_row_sha256": raw_hash,
                },
                final_lf=False,
            )
        )
        identities.append(
            {
                "source_row_number": number,
                "collection_id": fixture["collection_id"],
                "source_batch_id": row["source_batch_id"],
                "source_record_id": row["source_record_id"],
                "source_system": row["source_system"],
                "portfolio_id": row["portfolio_id"],
                "source_sha256": source_manifest["source_sha256"],
                "raw_row_sha256": raw_hash,
                "query_id": query_id,
                "opaque_batch_id": batch_id,
                "opaque_stratum_id": stratum_id,
            }
        )
        reasons: list[str] = []
        if counts[row["source_record_id"]] > 1:
            reasons.append("DUPLICATE_SOURCE_RECORD_ID")
        required = (
            "source_batch_id",
            "source_record_id",
            "source_system",
            "portfolio_id",
        )
        empty = {name for name in required if not row[name]}
        if empty:
            reasons.append("EMPTY_REQUIRED_PROVENANCE")
        mismatch = any(
            row[name] and row[name] != provenance[name]
            for name in ("source_batch_id", "source_system", "portfolio_id")
        )
        if mismatch:
            reasons.append("PROVENANCE_MISMATCH")
        has_name_or_address = bool(
            row["crm_name_raw"].strip() or row["crm_address_raw"].strip()
        )
        has_location = bool(
            row["crm_postcode_raw"].strip()
            or row["crm_city_raw"].strip()
            or row["crm_insee_raw"].strip()
        )
        if not (has_name_or_address and has_location):
            reasons.append("LOCATION_RULE_FAILED")
        if any(_has_decimal_leak(value) for value in values):
            reasons.append("UNICODE_DECIMAL_9_OR_14")
        reasons.sort(key=reason_order.index)
        for ordinal, reason in enumerate(reasons, start=1):
            reason_counts[reason] += 1
            proofs.append(
                {
                    "source_row_number": number,
                    "source_record_locator_sha256": locator,
                    "reason_code": reason,
                    "reason_ordinal": ordinal,
                    "raw_row_sha256": raw_hash,
                }
            )
        if not reasons:
            safe.append(
                {
                    "query_id": query_id,
                    "opaque_batch_id": batch_id,
                    "opaque_stratum_id": stratum_id,
                    "reference_date": fixture["reference_date"],
                    "crm_name_raw": row["crm_name_raw"],
                    "crm_address_raw": row["crm_address_raw"],
                    "crm_postcode_raw": row["crm_postcode_raw"],
                    "crm_city_raw": row["crm_city_raw"],
                    "crm_insee_raw": row["crm_insee_raw"],
                }
            )
    safe.sort(key=lambda value: value["query_id"])
    proofs.sort(
        key=lambda value: (
            value["source_row_number"],
            value["reason_ordinal"],
        )
    )
    identities.sort(key=lambda value: value["source_row_number"])
    return safe, proofs, identities, reason_counts


def _table_from_rows(
    schema_spec: Sequence[Sequence[Any]],
    rows: Sequence[Mapping[str, Any]],
) -> pa.Table:
    schema = pa.schema(
        [
            pa.field(name, _arrow_type(type_name), nullable=nullable)
            for name, type_name, nullable in schema_spec
        ]
    )
    arrays = [
        pa.array([row[field.name] for row in rows], type=field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _parquet_bytes(plan: Mapping[str, Any], table: pa.Table) -> bytes:
    if pa.__version__ != plan["parquet_writer"]["pyarrow"]:
        _stop("pyarrow runtime pin mismatch")
    output = io.BytesIO()
    writer = plan["parquet_writer"]
    pq.write_table(
        table,
        output,
        compression=writer["compression"],
        compression_level=writer["compression_level"],
        version=writer["format_version"],
        data_page_version=writer["data_page_version"],
        use_dictionary=writer["use_dictionary"],
        write_statistics=writer["write_statistics"],
        row_group_size=writer["row_group_size"],
        store_schema=writer["store_schema"],
    )
    return output.getvalue()


def _logical_rows_hash(
    schema_spec: Sequence[Sequence[Any]], rows: Sequence[Mapping[str, Any]]
) -> str:
    ordered = [
        {name: row[name] for name, _type, _nullable in schema_spec}
        for row in rows
    ]
    raw = json.dumps(
        ordered,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(raw)


def _build_scan_payloads(
    *,
    plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    rows: Sequence[Sequence[str]],
    run_id: str,
    attempt_id: str,
    control_hash: str,
    sealed: Mapping[str, Any],
    plan_hash: str,
) -> dict[str, bytes]:
    safe, proofs, identities, reason_counts = _scan_rows(
        plan, source_manifest, rows, run_id
    )
    for row in safe:
        if any(_has_decimal_leak(value) for value in row.values()):
            _stop("anti-leak rescan failed on safe query output")
    if not all(
        ID_RE.fullmatch(value)
        for row in identities
        for key, value in row.items()
        if key in {"query_id", "opaque_batch_id", "opaque_stratum_id"}
    ):
        _stop("opaque ID validation failed")
    schemas = plan["scan"]["tree"]["schemas"]
    logical_hashes: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    row_sets = {
        "safe_queries_preidentity.parquet": safe,
        "quarantine_proofs.parquet": proofs,
        "source_identity_map.parquet": identities,
    }
    for name, data_rows in row_sets.items():
        schema_spec = schemas[name]
        payloads[name] = _parquet_bytes(
            plan, _table_from_rows(schema_spec, data_rows)
        )
        logical_hashes[name] = _logical_rows_hash(schema_spec, data_rows)
    provenance = {
        "schema_version": "sireto-v4.12-fresh-synthetic-scan-provenance-1",
        "synthetic_run_id": run_id,
        "attempt_id": attempt_id,
        "logical_time_utc": plan["fixture"]["logical_time_utc"],
        "fixture_control_manifest_sha256": control_hash,
        "sealed_input_payload_manifest_sha256": sealed[
            "payload_manifest_sha256"
        ],
        "sealed_input_seal_sha256": sealed["seal_sha256"],
        "builder_source_sha256": _implementation_bundle_sha256(),
        "tests_sha256": _tests_sha256(),
        "plan_sha256": plan_hash,
    }
    provenance_bytes = canonical_json_bytes(provenance)
    logical_hashes["scan_provenance.json"] = sha256_bytes(
        canonical_json_bytes(provenance, final_lf=False)
    )
    integrity_projection = {
        "schema_version": "sireto-v4.12-fresh-synthetic-scan-integrity-1",
        "synthetic_run_id": run_id,
        "attempt_id": attempt_id,
        "collection_id": plan["fixture"]["collection_id"],
        "source_batch_id": plan["fixture"]["common_provenance"][
            "source_batch_id"
        ],
        "source_row_count": len(rows),
        "safe_row_count": len(safe),
        "quarantined_row_count": len(
            {proof["source_row_number"] for proof in proofs}
        ),
        "quarantine_proof_count": len(proofs),
        "reason_counts": reason_counts,
        "all_safe_output_strings_rescanned": True,
        "all_ids_pattern_valid": True,
    }
    logical_hashes["scan_integrity.json"] = sha256_bytes(
        canonical_json_bytes(integrity_projection, final_lf=False)
    )
    integrity = dict(integrity_projection)
    integrity["logical_hashes"] = logical_hashes
    payloads["scan_integrity.json"] = canonical_json_bytes(integrity)
    payloads["scan_provenance.json"] = provenance_bytes
    expected = plan["outputs"]["scan_output"]["payloads"]
    return {name: payloads[name] for name in expected}


def _parquet_rows_validated(
    plan: Mapping[str, Any],
    name: str,
    payload: bytes,
) -> list[dict[str, Any]]:
    schema_spec = plan["scan"]["tree"]["schemas"][name]
    expected_schema = pa.schema(
        [
            pa.field(field, _arrow_type(type_name), nullable=nullable)
            for field, type_name, nullable in schema_spec
        ]
    )
    table = pq.read_table(io.BytesIO(payload))
    parquet_file = pq.ParquetFile(io.BytesIO(payload))
    if (
        table.schema != expected_schema
        or table.schema.metadata not in (None, {})
        or parquet_file.schema_arrow.metadata not in (None, {})
    ):
        _stop(f"scan Parquet schema mismatch: {name}")
    file_metadata = parquet_file.metadata.metadata or {}
    allowed_file_metadata = (
        {b"ARROW:schema"} if plan["parquet_writer"]["store_schema"] else set()
    )
    if set(file_metadata) - allowed_file_metadata:
        _stop(f"scan Parquet application metadata forbidden: {name}")
    rows = table.to_pylist()
    if name == "safe_queries_preidentity.parquet":
        expected_order = sorted(rows, key=lambda row: row["query_id"])
    elif name == "quarantine_proofs.parquet":
        expected_order = sorted(
            rows,
            key=lambda row: (
                row["source_row_number"],
                row["reason_ordinal"],
            ),
        )
    else:
        expected_order = sorted(
            rows, key=lambda row: row["source_row_number"]
        )
    if rows != expected_order:
        _stop(f"scan Parquet sort mismatch: {name}")
    return rows


def _validate_scan_binding(
    *,
    plan: Mapping[str, Any],
    branch: Mapping[str, Any],
    sealed: Mapping[str, Any],
    run_id: str,
    attempt_id: str,
    control_hash: str,
    plan_hash: str,
) -> None:
    payloads = branch["payloads"]
    batch_reason, source_rows = _batch_parse(
        sealed["payloads"]["crm_safe.csv"]
    )
    if batch_reason is not None:
        _stop("scan output attached to batch-invalid source")
    source_manifest = json.loads(
        sealed["payloads"]["source_manifest.json"]
    )
    expected_payloads = _build_scan_payloads(
        plan=plan,
        source_manifest=source_manifest,
        rows=source_rows,
        run_id=run_id,
        attempt_id=attempt_id,
        control_hash=control_hash,
        sealed=sealed,
        plan_hash=plan_hash,
    )
    if payloads != expected_payloads:
        _stop("scan output differs from deterministic sealed-input rescan")
    rows_by_name = {
        name: _parquet_rows_validated(plan, name, payloads[name])
        for name in (
            "safe_queries_preidentity.parquet",
            "quarantine_proofs.parquet",
            "source_identity_map.parquet",
        )
    }
    provenance = json.loads(payloads["scan_provenance.json"])
    expected_provenance = {
        "schema_version": "sireto-v4.12-fresh-synthetic-scan-provenance-1",
        "synthetic_run_id": run_id,
        "attempt_id": attempt_id,
        "logical_time_utc": plan["fixture"]["logical_time_utc"],
        "fixture_control_manifest_sha256": control_hash,
        "sealed_input_payload_manifest_sha256": sealed[
            "payload_manifest_sha256"
        ],
        "sealed_input_seal_sha256": sealed["seal_sha256"],
        "builder_source_sha256": _implementation_bundle_sha256(),
        "tests_sha256": _tests_sha256(),
        "plan_sha256": plan_hash,
    }
    if (
        canonical_json_bytes(provenance)
        != payloads["scan_provenance.json"]
        or provenance != expected_provenance
    ):
        _stop("scan provenance input/code/test binding mismatch")
    integrity = json.loads(payloads["scan_integrity.json"])
    if canonical_json_bytes(integrity) != payloads["scan_integrity.json"]:
        _stop("scan integrity JSON is not canonical")
    proofs = rows_by_name["quarantine_proofs.parquet"]
    safe = rows_by_name["safe_queries_preidentity.parquet"]
    identities = rows_by_name["source_identity_map.parquet"]
    reason_counts = {
        reason: 0
        for reason in plan["scan"]["typed_json"]["reason_counts"]["keys"]
    }
    for proof in proofs:
        if proof["reason_code"] not in reason_counts:
            _stop("unknown row quarantine reason")
        reason_counts[proof["reason_code"]] += 1
    projection = {
        "schema_version": "sireto-v4.12-fresh-synthetic-scan-integrity-1",
        "synthetic_run_id": run_id,
        "attempt_id": attempt_id,
        "collection_id": plan["fixture"]["collection_id"],
        "source_batch_id": plan["fixture"]["common_provenance"][
            "source_batch_id"
        ],
        "source_row_count": len(identities),
        "safe_row_count": len(safe),
        "quarantined_row_count": len(
            {proof["source_row_number"] for proof in proofs}
        ),
        "quarantine_proof_count": len(proofs),
        "reason_counts": reason_counts,
        "all_safe_output_strings_rescanned": True,
        "all_ids_pattern_valid": True,
    }
    logical_hashes = {
        name: _logical_rows_hash(
            plan["scan"]["tree"]["schemas"][name], rows
        )
        for name, rows in rows_by_name.items()
    }
    logical_hashes["scan_integrity.json"] = sha256_bytes(
        canonical_json_bytes(projection, final_lf=False)
    )
    logical_hashes["scan_provenance.json"] = sha256_bytes(
        canonical_json_bytes(provenance, final_lf=False)
    )
    expected_integrity = dict(projection)
    expected_integrity["logical_hashes"] = logical_hashes
    if integrity != expected_integrity:
        _stop("scan integrity counts or logical hashes mismatch")
    if len(identities) != plan["fixture"]["expected"]["source_row_count"]:
        _stop("scan identity-map denominator mismatch")
    safe_query_ids = {row["query_id"] for row in safe}
    identity_query_ids = {row["query_id"] for row in identities}
    if not safe_query_ids <= identity_query_ids:
        _stop("safe query is absent from private identity map")
    for row in safe:
        if any(_has_decimal_leak(value) for value in row.values()):
            _stop("safe output anti-leak validation failed")


def _validate_quarantine_binding(
    *,
    plan: Mapping[str, Any],
    branch: Mapping[str, Any],
    sealed: Mapping[str, Any],
    run_id: str,
    observed_reason: str,
) -> None:
    raw = branch["payloads"]["batch_quarantine_proof.json"]
    proof = json.loads(raw)
    expected = {
        "schema_version": (
            "sireto-v4.12-fresh-synthetic-batch-quarantine-proof-1"
        ),
        "synthetic_run_id": run_id,
        "collection_id": plan["fixture"]["collection_id"],
        "source_batch_id": plan["fixture"]["common_provenance"][
            "source_batch_id"
        ],
        "reason_code": observed_reason,
        "expected_source_row_count": plan["fixture"]["expected"][
            "source_row_count"
        ],
        "sealed_input_payload_manifest_sha256": sealed[
            "payload_manifest_sha256"
        ],
        "sealed_input_seal_sha256": sealed["seal_sha256"],
        "sealed_source_sha256": sha256_bytes(
            sealed["payloads"]["crm_safe.csv"]
        ),
        "logical_time_utc": plan["fixture"]["logical_time_utc"],
    }
    if canonical_json_bytes(proof) != raw or proof != expected:
        _stop("batch quarantine proof input/reason binding mismatch")


def _receipt_id(
    plan: Mapping[str, Any], kind: str, payload: Mapping[str, Any]
) -> str:
    return opaque_digest(plan["ids"][kind]["domain"], payload)


def _receipt_paths(run_id: str) -> tuple[str, str]:
    audit = f"audit/{run_id}"
    return f"{audit}/receipts/collections", f"{audit}/receipts/batches"


def _create_receipts(
    *,
    plan: Mapping[str, Any],
    authority: _RootFD,
    root: Path,
    run_id: str,
    sealed_path: str,
    sealed: Mapping[str, Any],
    audit_time: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads = sealed["payloads"]
    fixture = plan["fixture"]
    collection_hash = sha256_bytes(
        payloads["collection_source_manifest.json"]
    )
    source_hash = sha256_bytes(payloads["source_manifest.json"])
    evidence_hash = sha256_bytes(
        payloads["evidence_source_manifest.json"]
    )
    csv_hash = sha256_bytes(payloads["crm_safe.csv"])
    collection_id = _receipt_id(
        plan,
        "collection_receipt",
        {
            "synthetic_run_id": run_id,
            "collection_id": fixture["collection_id"],
            "collection_manifest_sha256": collection_hash,
            "payload_manifest_sha256": sealed["payload_manifest_sha256"],
            "seal_sha256": sealed["seal_sha256"],
        },
    )
    batch_id = _receipt_id(
        plan,
        "batch_receipt",
        {
            "synthetic_run_id": run_id,
            "collection_id": fixture["collection_id"],
            "source_batch_id": fixture["common_provenance"][
                "source_batch_id"
            ],
            "sealed_source_sha256": csv_hash,
            "payload_manifest_sha256": sealed["payload_manifest_sha256"],
            "seal_sha256": sealed["seal_sha256"],
        },
    )
    collection = {
        "schema_version": "sireto-v4.12-fresh-synthetic-collection-receipt-1",
        "receipt_kind": "COLLECTION",
        "receipt_id": collection_id,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "sealed_tree_absolute_path": str(root / sealed_path),
        "collection_manifest_relative_path": "collection_source_manifest.json",
        "collection_manifest_size_bytes": len(
            payloads["collection_source_manifest.json"]
        ),
        "collection_manifest_sha256": collection_hash,
        "payload_manifest_sha256": sealed["payload_manifest_sha256"],
        "seal_sha256": sealed["seal_sha256"],
        "created_at_utc": audit_time,
    }
    batch = {
        "schema_version": "sireto-v4.12-fresh-synthetic-batch-receipt-1",
        "receipt_kind": "BATCH",
        "receipt_id": batch_id,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "source_batch_id": fixture["common_provenance"]["source_batch_id"],
        "sealed_tree_absolute_path": str(root / sealed_path),
        "sealed_source_relative_path": "crm_safe.csv",
        "sealed_source_size_bytes": len(payloads["crm_safe.csv"]),
        "sealed_source_sha256": csv_hash,
        "source_manifest_sha256": source_hash,
        "evidence_manifest_sha256": evidence_hash,
        "payload_manifest_sha256": sealed["payload_manifest_sha256"],
        "seal_sha256": sealed["seal_sha256"],
        "created_at_utc": audit_time,
    }
    collection_root, batch_root = _receipt_paths(run_id)
    for relative in (collection_root, batch_root):
        fd = authority.open_dir(relative, create=True)
        os.close(fd)
    for relative, expected_id in (
        (collection_root, collection_id),
        (batch_root, batch_id),
    ):
        siblings = authority.list(relative)
        if siblings not in ([], [expected_id]):
            _stop("receipt root contains an unexpected sibling")
    collection_dir = f"{collection_root}/{collection_id}"
    batch_dir = f"{batch_root}/{batch_id}"
    if not authority.exists(collection_dir, directory=True):
        authority.mkdir_exclusive(collection_dir)
        authority.write_exclusive(
            f"{collection_dir}/receipt.json", canonical_json_bytes(collection)
        )
    else:
        entries = authority.list(collection_dir)
        if entries == []:
            authority.write_exclusive(
                f"{collection_dir}/receipt.json",
                canonical_json_bytes(collection),
            )
        elif entries != ["receipt.json"]:
            _stop("collection receipt directory exact-tree mismatch")
        existing, _raw = _decode_json_canonical(
            authority.read_file(f"{collection_dir}/receipt.json"),
            f"{collection_dir}/receipt.json",
        )
        if (
            set(existing) != set(plan["receipts"]["collection_exact_fields"])
            or not _is_strict_rfc3339_utc_seconds(
                existing["created_at_utc"]
            )
        ):
            _stop("collection receipt exact schema mismatch")
        if {
            key: value
            for key, value in existing.items()
            if key != "created_at_utc"
        } != {
            key: value
            for key, value in collection.items()
            if key != "created_at_utc"
        }:
            _stop("collection receipt conflict")
        collection = existing
    if not authority.exists(batch_dir, directory=True):
        authority.mkdir_exclusive(batch_dir)
        authority.write_exclusive(
            f"{batch_dir}/receipt.json", canonical_json_bytes(batch)
        )
    else:
        entries = authority.list(batch_dir)
        if entries == []:
            authority.write_exclusive(
                f"{batch_dir}/receipt.json", canonical_json_bytes(batch)
            )
        elif entries != ["receipt.json"]:
            _stop("batch receipt directory exact-tree mismatch")
        existing, _raw = _decode_json_canonical(
            authority.read_file(f"{batch_dir}/receipt.json"),
            f"{batch_dir}/receipt.json",
        )
        if (
            set(existing) != set(plan["receipts"]["batch_exact_fields"])
            or not _is_strict_rfc3339_utc_seconds(
                existing["created_at_utc"]
            )
        ):
            _stop("batch receipt exact schema mismatch")
        if {
            key: value
            for key, value in existing.items()
            if key != "created_at_utc"
        } != {
            key: value
            for key, value in batch.items()
            if key != "created_at_utc"
        }:
            _stop("batch receipt conflict")
        batch = existing
    if authority.list(collection_root) != [collection_id]:
        _stop("collection receipt root final exact-tree mismatch")
    if authority.list(batch_root) != [batch_id]:
        _stop("batch receipt root final exact-tree mismatch")
    return collection, batch


def _event_dirs(authority: _RootFD, run_id: str) -> tuple[str, str]:
    audit = f"audit/{run_id}"
    events = f"{audit}/events"
    generations = f"{audit}/events_manifests"
    for relative in (events, generations):
        fd = authority.open_dir(relative, create=True)
        os.close(fd)
    return events, generations


def _validate_event_shape(
    plan: Mapping[str, Any],
    event: Mapping[str, Any],
    sequence: int,
    run_id: str,
) -> None:
    fixture = plan["fixture"]
    is_batch = event.get("entity_kind") == "BATCH"
    if (
        set(event) != set(plan["events"]["event_exact_fields"])
        or event["schema_version"] != plan["events"]["event_schema"]
        or event["synthetic_run_id"] != run_id
        or event["sequence"] != sequence
        or event["collection_id"] != fixture["collection_id"]
        or event["entity_id"]
        != (
            fixture["common_provenance"]["source_batch_id"]
            if is_batch
            else fixture["collection_id"]
        )
        or event["source_batch_id"]
        != (
            fixture["common_provenance"]["source_batch_id"]
            if is_batch
            else None
        )
        or any(value is None for value in event["manifest_hashes"].values())
        or not _is_strict_rfc3339_utc_seconds(event["timestamp_utc"])
        or set(event["manifest_hashes"])
        != set(plan["events"]["event_map_schema"]["manifest_hashes"])
        or set(event["tree_hashes"])
        != set(plan["events"]["event_map_schema"]["tree_hashes"])
        or any(
            value is not None
            and (
                type(value) is not str
                or HEX_RE.fullmatch(value) is None
            )
            for value in event["tree_hashes"].values()
        )
        or any(
            value is not None
            and (
                type(value) is not str
                or HEX_RE.fullmatch(value) is None
            )
            for value in event["manifest_hashes"].values()
        )
    ):
        _stop("event exact schema or typed maps mismatch")
    if sequence == 1:
        valid = (
            event["entity_kind"] == "BATCH"
            and event["previous_state"] == "WAITING_STABLE"
            and event["new_state"] == "RECEIPTED"
            and event["previous_event_sha256"] is None
            and all(
                event["tree_hashes"][key] is not None
                for key in (
                    "sealed_input_payload_manifest_sha256",
                    "sealed_input_seal_sha256",
                )
            )
            and all(
                event["tree_hashes"][key] is None
                for key in (
                    "scan_output_payload_manifest_sha256",
                    "scan_output_seal_sha256",
                    "batch_quarantine_payload_manifest_sha256",
                    "batch_quarantine_seal_sha256",
                )
            )
        )
    elif sequence == 2:
        tree = event["tree_hashes"]
        branch_valid = (
            event["new_state"] == "INGESTED"
            and tree["scan_output_payload_manifest_sha256"] is not None
            and tree["scan_output_seal_sha256"] is not None
            and tree["batch_quarantine_payload_manifest_sha256"] is None
            and tree["batch_quarantine_seal_sha256"] is None
        ) or (
            event["new_state"] == "QUARANTINED"
            and tree["scan_output_payload_manifest_sha256"] is None
            and tree["scan_output_seal_sha256"] is None
            and tree["batch_quarantine_payload_manifest_sha256"] is not None
            and tree["batch_quarantine_seal_sha256"] is not None
        ) or event["new_state"] == "STOPPED"
        valid = (
            event["entity_kind"] == "BATCH"
            and event["previous_state"] == "RECEIPTED"
            and event["new_state"] in {"INGESTED", "QUARANTINED", "STOPPED"}
            and event["previous_event_sha256"] is not None
            and branch_valid
        )
    else:
        valid = (
            event["entity_kind"] == "COLLECTION"
            and event["source_batch_id"] is None
            and event["previous_state"] == "WAITING"
            and event["new_state"] in {"INGESTED", "QUARANTINED", "STOPPED"}
            and event["previous_event_sha256"] is not None
        )
    if not valid:
        _stop("event state-machine semantics mismatch")


def _validate_journal(
    plan: Mapping[str, Any], authority: _RootFD, run_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    events_dir, generations_dir = _event_dirs(authority, run_id)
    generation_names = authority.list(generations_dir)
    events: list[dict[str, Any]] = []
    event_hashes: list[str] = []
    previous_event: str | None = None
    previous_generation: str | None = None
    prior_records: list[dict[str, Any]] = []
    for sequence, generation_name in enumerate(generation_names, start=1):
        generation, generation_raw = _decode_json_canonical(
            authority.read_file(f"{generations_dir}/{generation_name}"),
            generation_name,
        )
        generation_hash = sha256_bytes(generation_raw)
        if generation_name != f"{sequence:08d}-{generation_hash}.json":
            _stop("generation path/hash conflict")
        if (
            set(generation) != set(plan["events"]["generation_exact_fields"])
            or generation["schema_version"]
            != "sireto-v4.12-fresh-synthetic-scanner-sealer-event-generation-1"
            or generation["generation"] != sequence
            or not _is_strict_rfc3339_utc_seconds(
                generation["created_at_utc"]
            )
            or generation["event_count"] != sequence
            or generation["previous_manifest_sha256"] != previous_generation
            or len(generation["ordered_event_records"]) != sequence
        ):
            _stop("generation chain conflict")
        records = generation["ordered_event_records"]
        if records[:-1] != prior_records:
            _stop("generation is not the cumulative prior prefix")
        record = records[-1]
        if set(record) != set(plan["events"]["manifest_record_exact_fields"]):
            _stop("generation event record fields mismatch")
        event_name = Path(record["relative_path"]).name
        if (
            record["relative_path"] != f"events/{event_name}"
            or "/" in event_name
            or record["sequence"] != sequence
            or event_name != f"{sequence:08d}-{record['sha256']}.json"
        ):
            _stop("generation event record path conflict")
        event_relative = f"{events_dir}/{event_name}"
        if not authority.exists(event_relative, directory=False):
            _stop("generation references a missing event")
        event, event_raw = _decode_json_canonical(
            authority.read_file(event_relative), event_relative
        )
        event_hash = sha256_bytes(event_raw)
        if (
            record["size_bytes"] != len(event_raw)
            or record["sha256"] != event_hash
            or event["sequence"] != sequence
            or event["previous_event_sha256"] != previous_event
            or generation["head_event_sha256"] != event_hash
        ):
            _stop("referenced event chain conflict")
        _validate_event_shape(plan, event, sequence, run_id)
        if sequence == 3 and (
            event["new_state"] != events[1]["new_state"]
            or event["manifest_hashes"] != events[1]["manifest_hashes"]
            or event["tree_hashes"] != events[1]["tree_hashes"]
        ):
            _stop("collection event does not mirror batch terminal event")
        events.append(event)
        event_hashes.append(event_hash)
        prior_records = records
        previous_event = event_hash
        previous_generation = generation_hash
    return events, event_hashes


def _append_event(
    *,
    plan: Mapping[str, Any],
    authority: _RootFD,
    run_id: str,
    event: Mapping[str, Any],
    audit_time: str,
) -> dict[str, Any]:
    events, hashes = _validate_journal(plan, authority, run_id)
    sequence = len(events) + 1
    if sequence > 3:
        _stop("journal already terminal")
    payload = dict(event)
    payload.update(
        {
            "schema_version": plan["events"]["event_schema"],
            "synthetic_run_id": run_id,
            "sequence": sequence,
            "previous_event_sha256": hashes[-1] if hashes else None,
            "timestamp_utc": audit_time,
        }
    )
    if set(payload) != set(plan["events"]["event_exact_fields"]):
        _stop("event fields mismatch")
    event_raw = canonical_json_bytes(payload)
    event_hash = sha256_bytes(event_raw)
    events_dir, generations_dir = _event_dirs(authority, run_id)
    event_name = f"{sequence:08d}-{event_hash}.json"
    authority.write_exclusive(f"{events_dir}/{event_name}", event_raw)
    records = []
    for index, digest in enumerate([*hashes, event_hash], start=1):
        name = f"{index:08d}-{digest}.json"
        raw = authority.read_file(f"{events_dir}/{name}")
        records.append(
            {
                "relative_path": f"events/{name}",
                "sequence": index,
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    previous_manifest = None
    existing_generations = authority.list(generations_dir)
    if existing_generations:
        previous_manifest = sha256_bytes(
            authority.read_file(
                f"{generations_dir}/{existing_generations[-1]}"
            )
        )
    generation = {
        "schema_version": (
            "sireto-v4.12-fresh-synthetic-scanner-sealer-event-generation-1"
        ),
        "generation": sequence,
        "ordered_event_records": records,
        "event_count": sequence,
        "head_event_sha256": event_hash,
        "previous_manifest_sha256": previous_manifest,
        "created_at_utc": audit_time,
    }
    generation_raw = canonical_json_bytes(generation)
    generation_hash = sha256_bytes(generation_raw)
    authority.write_exclusive(
        f"{generations_dir}/{sequence:08d}-{generation_hash}.json",
        generation_raw,
    )
    _validate_journal(plan, authority, run_id)
    return payload


def _hash_maps(
    source_payloads: Mapping[str, bytes],
    *,
    sealed_input: Mapping[str, Any],
    scan: Mapping[str, Any] | None = None,
    quarantine: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = {
        "collection_source_manifest_sha256": sha256_bytes(
            source_payloads["collection_source_manifest.json"]
        ),
        "evidence_source_manifest_sha256": sha256_bytes(
            source_payloads["evidence_source_manifest.json"]
        ),
        "source_manifest_sha256": sha256_bytes(
            source_payloads["source_manifest.json"]
        ),
    }
    trees = {
        "batch_quarantine_payload_manifest_sha256": (
            quarantine["payload_manifest_sha256"] if quarantine else None
        ),
        "batch_quarantine_seal_sha256": (
            quarantine["seal_sha256"] if quarantine else None
        ),
        "scan_output_payload_manifest_sha256": (
            scan["payload_manifest_sha256"] if scan else None
        ),
        "scan_output_seal_sha256": scan["seal_sha256"] if scan else None,
        "sealed_input_payload_manifest_sha256": sealed_input[
            "payload_manifest_sha256"
        ],
        "sealed_input_seal_sha256": sealed_input["seal_sha256"],
    }
    return manifests, trees


def _event_base(
    plan: Mapping[str, Any],
    *,
    entity_kind: str,
    previous_state: str,
    new_state: str,
    manifest_hashes: Mapping[str, Any],
    tree_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    fixture = plan["fixture"]
    is_batch = entity_kind == "BATCH"
    return {
        "entity_kind": entity_kind,
        "entity_id": (
            fixture["common_provenance"]["source_batch_id"]
            if is_batch
            else fixture["collection_id"]
        ),
        "collection_id": fixture["collection_id"],
        "source_batch_id": (
            fixture["common_provenance"]["source_batch_id"]
            if is_batch
            else None
        ),
        "previous_state": previous_state,
        "new_state": new_state,
        "manifest_hashes": dict(manifest_hashes),
        "tree_hashes": dict(tree_hashes),
    }


def _quarantine_payload(
    plan: Mapping[str, Any],
    run_id: str,
    reason: str,
    sealed: Mapping[str, Any],
) -> dict[str, bytes]:
    proof = {
        "schema_version": (
            "sireto-v4.12-fresh-synthetic-batch-quarantine-proof-1"
        ),
        "synthetic_run_id": run_id,
        "collection_id": plan["fixture"]["collection_id"],
        "source_batch_id": plan["fixture"]["common_provenance"][
            "source_batch_id"
        ],
        "reason_code": reason,
        "expected_source_row_count": plan["fixture"]["expected"][
            "source_row_count"
        ],
        "sealed_input_payload_manifest_sha256": sealed[
            "payload_manifest_sha256"
        ],
        "sealed_input_seal_sha256": sealed["seal_sha256"],
        "sealed_source_sha256": sha256_bytes(
            sealed["payloads"]["crm_safe.csv"]
        ),
        "logical_time_utc": plan["fixture"]["logical_time_utc"],
    }
    return {"batch_quarantine_proof.json": canonical_json_bytes(proof)}


def run_scanner(
    plan_path: Path,
    control_manifest_path: Path,
    root: Path,
    *,
    stability_wait_seconds: float = 60.0,
    sleep_fn: Callable[[float], Any] = time.sleep,
    audit_now_fn: Callable[[], Any] = _audit_now,
    _test_mode: bool = False,
    _test_fault_fn: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Scan, seal and journal the S0 fixture under an injected synthetic root."""

    if _test_mode is not True:
        _stop("run_scanner is disabled until lock and sandbox are implemented")
    pytest_root = Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/pytest_v412"
    )
    requested_root = Path(os.path.abspath(os.fspath(root)))
    if (
        "PYTEST_CURRENT_TEST" not in os.environ
        or requested_root == pytest_root
        or not requested_root.is_relative_to(pytest_root)
    ):
        _stop(
            "test mode requires PYTEST_CURRENT_TEST and a root strictly "
            "below /Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/pytest_v412"
        )
    if (
        not isinstance(stability_wait_seconds, (int, float))
        or not math.isfinite(stability_wait_seconds)
        or stability_wait_seconds < 0
    ):
        _stop("test stability interval must be finite and non-negative")
    old_umask = os.umask(0o077)
    authority: _RootFD | None = None
    plan: dict[str, Any] | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    sealed: dict[str, Any] | None = None
    try:
        plan, plan_raw = _load_plan(Path(plan_path))
        root = _validate_root(Path(root))
        authority = _RootFD(root)
        authority.require_beneath(pytest_root)
        control_path = Path(control_manifest_path)
        control_absolute = Path(os.path.abspath(os.fspath(control_path)))
        try:
            control_relative = control_absolute.relative_to(root).as_posix()
        except ValueError:
            _stop("control manifest is outside the synthetic root")
        control, control_raw = _decode_json_canonical(
            authority.read_file(control_relative), control_relative
        )
        if set(control) != set(plan["control_manifest"]["exact_fields"]):
            _stop("control manifest fields mismatch")
        if control["schema_version"] != plan["control_manifest"]["schema"]:
            _stop("control manifest schema mismatch")
        if (
            control["synthetic_fixture"] is not True
            or control["fixture_spec_sha256"]
            != plan["control_manifest"]["fixture_spec_sha256"]
            or control["logical_time_utc"]
            != plan["fixture"]["logical_time_utc"]
            or control["batch_count"] != 1
            or control["expected_source_row_count"] != 6
            or control["producer_exclusions"] != []
            or any(
                HEX_RE.fullmatch(control[field]) is None
                for field in (
                    "fixture_spec_sha256",
                    "collection_source_manifest_sha256",
                    "source_manifest_sha256",
                    "crm_safe_csv_sha256",
                    "evidence_source_manifest_sha256",
                    "evidence_source_parquet_sha256",
                )
            )
        ):
            _stop("control manifest constants or types mismatch")
        plan_hash = sha256_bytes(plan_raw)
        expected_run = opaque_digest(
            plan["ids"]["run"]["domain"],
            {
                "fixture_spec_sha256": plan["control_manifest"][
                    "fixture_spec_sha256"
                ],
                "plan_sha256": plan_hash,
            },
        )
        run_id = control["synthetic_run_id"]
        if run_id != expected_run or not ID_RE.fullmatch(run_id):
            _stop("synthetic run ID mismatch")
        expected_control = (
            f"control/{run_id}/fixture_control_manifest.json"
        )
        if control_relative != expected_control:
            _stop("control manifest is outside the synthetic run")
        control_hash = sha256_bytes(control_raw)
        attempt_id = opaque_digest(
            plan["ids"]["attempt"]["domain"],
            {
                "synthetic_run_id": run_id,
                "fixture_control_manifest_sha256": control_hash,
                "logical_time_utc": control["logical_time_utc"],
            },
        )
        audit_time = _audit_timestamp(audit_now_fn())
        sealed_path = f"sealed/{run_id}/input"
        if authority.exists(sealed_path, directory=True):
            sealed = _validate_tree(
                authority,
                sealed_path,
                package_kind="SEALED_INPUT",
                expected_payload_names=plan["outputs"]["sealed_input"][
                    "payloads"
                ],
                plan=plan,
                run_id=run_id,
            )
        else:
            package = f"inbox/{run_id}/package"
            payloads = _open_stable_payloads(
                authority,
                package,
                plan["input_package"]["exact_files"],
                stability_wait_seconds,
                sleep_fn,
            )
            _validate_source_manifests(
                plan, control, payloads, run_id
            )
            sealed = _seal_tree(
                plan=plan,
                authority=authority,
                run_id=run_id,
                destination=sealed_path,
                package_kind="SEALED_INPUT",
                payloads=payloads,
            )
            sealed = _validate_tree(
                authority,
                sealed_path,
                package_kind="SEALED_INPUT",
                expected_payload_names=plan["outputs"]["sealed_input"][
                    "payloads"
                ],
                plan=plan,
                run_id=run_id,
            )
        collection_manifest, source_manifest, _evidence = (
            _validate_source_manifests(
                plan, control, sealed["payloads"], run_id
            )
        )
        _create_receipts(
            plan=plan,
            authority=authority,
            root=root,
            run_id=run_id,
            sealed_path=sealed_path,
            sealed=sealed,
            audit_time=audit_time,
        )
        events, _hashes = _validate_journal(plan, authority, run_id)
        manifests, trees = _hash_maps(
            sealed["payloads"], sealed_input=sealed
        )
        if not events:
            _append_event(
                plan=plan,
                authority=authority,
                run_id=run_id,
                event=_event_base(
                    plan,
                    entity_kind="BATCH",
                    previous_state="WAITING_STABLE",
                    new_state="RECEIPTED",
                    manifest_hashes=manifests,
                    tree_hashes=trees,
                ),
                audit_time=audit_time,
            )
            events, _hashes = _validate_journal(plan, authority, run_id)
        if events[0]["new_state"] != "RECEIPTED":
            _stop("invalid seq1 state")
        if (
            events[0]["manifest_hashes"] != manifests
            or events[0]["tree_hashes"] != trees
        ):
            _stop("seq1 does not bind the validated sealed input")
        if _test_fault_fn is not None:
            try:
                _test_fault_fn("AFTER_SEQ1")
            except Exception as exc:
                _stop(f"test fault AFTER_SEQ1: {exc}")

        batch_reason, rows = _batch_parse(sealed["payloads"]["crm_safe.csv"])
        scan_path = f"scan/{run_id}/output"
        quarantine_path = f"quarantine/{run_id}/batch"
        branch: Mapping[str, Any]
        terminal_state: str
        if batch_reason is not None:
            if authority.exists(scan_path, directory=True):
                _stop("scan tree conflicts with batch quarantine")
            if not authority.exists(quarantine_path, directory=True):
                _seal_tree(
                    plan=plan,
                    authority=authority,
                    run_id=run_id,
                    destination=quarantine_path,
                    package_kind="BATCH_QUARANTINE",
                    payloads=_quarantine_payload(
                        plan, run_id, batch_reason, sealed
                    ),
                )
            branch = _validate_tree(
                authority,
                quarantine_path,
                package_kind="BATCH_QUARANTINE",
                expected_payload_names=plan["outputs"]["batch_quarantine"][
                    "payloads"
                ],
                plan=plan,
                run_id=run_id,
            )
            _validate_quarantine_binding(
                plan=plan,
                branch=branch,
                sealed=sealed,
                run_id=run_id,
                observed_reason=batch_reason,
            )
            terminal_state = "QUARANTINED"
            manifests, trees = _hash_maps(
                sealed["payloads"],
                sealed_input=sealed,
                quarantine=branch,
            )
        else:
            if authority.exists(quarantine_path, directory=True):
                _stop("batch quarantine conflicts with valid scan")
            if len(rows) != control["expected_source_row_count"]:
                _stop("source row count differs from control")
            if not authority.exists(scan_path, directory=True):
                payloads = _build_scan_payloads(
                    plan=plan,
                    source_manifest=source_manifest,
                    rows=rows,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    control_hash=control_hash,
                    sealed=sealed,
                    plan_hash=plan_hash,
                )
                _seal_tree(
                    plan=plan,
                    authority=authority,
                    run_id=run_id,
                    destination=scan_path,
                    package_kind="SCAN_OUTPUT",
                    payloads=payloads,
                )
            branch = _validate_tree(
                authority,
                scan_path,
                package_kind="SCAN_OUTPUT",
                expected_payload_names=plan["outputs"]["scan_output"][
                    "payloads"
                ],
                plan=plan,
                run_id=run_id,
            )
            _validate_scan_binding(
                plan=plan,
                branch=branch,
                sealed=sealed,
                run_id=run_id,
                attempt_id=attempt_id,
                control_hash=control_hash,
                plan_hash=plan_hash,
            )
            terminal_state = "INGESTED"
            manifests, trees = _hash_maps(
                sealed["payloads"], sealed_input=sealed, scan=branch
            )
        events, _hashes = _validate_journal(plan, authority, run_id)
        if len(events) == 1:
            _append_event(
                plan=plan,
                authority=authority,
                run_id=run_id,
                event=_event_base(
                    plan,
                    entity_kind="BATCH",
                    previous_state="RECEIPTED",
                    new_state=terminal_state,
                    manifest_hashes=manifests,
                    tree_hashes=trees,
                ),
                audit_time=audit_time,
            )
            events, _hashes = _validate_journal(plan, authority, run_id)
        if events[1]["new_state"] != terminal_state:
            _stop("seq2 terminal state conflicts with validated branch")
        if (
            events[1]["manifest_hashes"] != manifests
            or events[1]["tree_hashes"] != trees
        ):
            _stop("seq2 does not bind the validated terminal tree")
        if len(events) == 2:
            _append_event(
                plan=plan,
                authority=authority,
                run_id=run_id,
                event=_event_base(
                    plan,
                    entity_kind="COLLECTION",
                    previous_state="WAITING",
                    new_state=terminal_state,
                    manifest_hashes=manifests,
                    tree_hashes=trees,
                ),
                audit_time=audit_time,
            )
            events, _hashes = _validate_journal(plan, authority, run_id)
        if len(events) != 3 or events[2]["new_state"] != terminal_state:
            _stop("collection terminal event conflict")
        if (
            events[2]["manifest_hashes"] != manifests
            or events[2]["tree_hashes"] != trees
        ):
            _stop("seq3 does not bind the validated terminal tree")
        verdict = (
            plan["verdicts"]["ingested"]
            if terminal_state == "INGESTED"
            else plan["verdicts"]["quarantined"]
        )
        return {
            "verdict": verdict,
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "sealed_input_path": str(root / sealed_path),
            "terminal_tree_path": str(
                root
                / (
                    scan_path
                    if terminal_state == "INGESTED"
                    else quarantine_path
                )
            ),
            "event_count": 3,
        }
    except (ScannerStop, OSError, ValueError, UnicodeError, csv.Error) as caught:
        original = (
            caught
            if isinstance(caught, ScannerStop)
            else ScannerStop(
                f"STOP {type(caught).__name__}: {caught}"
            )
        )
        if (
            authority is not None
            and plan is not None
            and run_id is not None
            and attempt_id is not None
            and sealed is not None
        ):
            try:
                events, _hashes = _validate_journal(
                    plan, authority, run_id
                )
            except ScannerStop:
                raise original
            if len(events) == 1 and events[0]["new_state"] == "RECEIPTED":
                audit_time = _audit_timestamp(audit_now_fn())
                manifests, trees = _hash_maps(
                    sealed["payloads"], sealed_input=sealed
                )
                _append_event(
                    plan=plan,
                    authority=authority,
                    run_id=run_id,
                    event=_event_base(
                        plan,
                        entity_kind="BATCH",
                        previous_state="RECEIPTED",
                        new_state="STOPPED",
                        manifest_hashes=manifests,
                        tree_hashes=trees,
                    ),
                    audit_time=audit_time,
                )
                _append_event(
                    plan=plan,
                    authority=authority,
                    run_id=run_id,
                    event=_event_base(
                        plan,
                        entity_kind="COLLECTION",
                        previous_state="WAITING",
                        new_state="STOPPED",
                        manifest_hashes=manifests,
                        tree_hashes=trees,
                    ),
                    audit_time=audit_time,
                )
                return {
                    "verdict": plan["verdicts"]["stop"],
                    "reason": str(original),
                    "synthetic_run_id": run_id,
                    "attempt_id": attempt_id,
                    "event_count": 3,
                }
        if original is caught:
            raise original
        raise original from caught
    finally:
        if authority is not None:
            authority.close()
        os.umask(old_umask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.parse_args()
    _stop(
        "production CLI disabled until the pinned lock and deny-default "
        "sandbox launcher are implemented"
    )


if __name__ == "__main__":
    raise SystemExit(main())
