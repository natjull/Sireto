#!/usr/bin/env python3
"""Build the frozen V4.12-R30 REVIEW docket without opening any label."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "sireto-v4.12-r30-docket-1"
SELECTION_DOMAIN = "SIRETO-V412-R30-SELECTION\0"
STRATUM_ORDER = (
    "SAME_SIREN_MULTISITE",
    "CROSS_SIREN_COLLISION",
    "OTHER_REVIEW",
)
EXPECTED_SELECTION_SHA256 = (
    "ec481d8db07165185fecc61bf437d868bfcbe4db6f4938a62b6c344e7000c2ee"
)
EXPECTED_COUNTS = {
    "query_count": 1_456,
    "auto_match_count": 1_177,
    "review_count": 279,
    "SAME_SIREN_MULTISITE": 40,
    "CROSS_SIREN_COLLISION": 199,
    "OTHER_REVIEW": 40,
}
FORBIDDEN_PARTITIONS = frozenset(
    {"random_sealed", "hard_dev_locked", "descriptive_locked"}
)
FORBIDDEN_ROLES = frozenset(
    {
        "random_sealed",
        "historical_random_excluded",
        "hard_dev_locked",
        "descriptive_locked",
    }
)
CONTRACT_SHA256 = (
    "f594800f4011ebf243987c36f31dd03d425f59d01226b03a1ddc2c11806592cc"
)
CANONICAL_CONTRACT_PATH = (
    Path(__file__).absolute().parent.parent
    / "docs/v4_12_review_adjudication_pilot_contract.md"
)

REFERENCE_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/"
    "v4_12_service_parity/b4b7fef24c5e7036"
)
PARTITION_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_8_acceptor_partitions/1c78764d5263afca"
)
ADJUDICATION_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "v4_7_current_adjudications/4cc5420fb5da0683"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "v4_12_review_adjudication_pilot"
)
FUTURE_SIRENE_SNAPSHOT = {
    "path": (
        "/Users/nathanjullia/Documents/Projets/SIRETO/"
        "data/StockEtablissement_utf8.parquet"
    ),
    "sha256": "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845",
    "row_count": 42_322_035,
    "opened_by_builder": False,
}
TEST_SOURCE = Path(__file__).absolute().parent.parent / (
    "tests/test_build_v412_review_adjudication_pilot.py"
)

CANONICAL_INPUTS = {
    "reference_manifest": (
        REFERENCE_ROOT / "manifest.json",
        "cbcb3303107cd00f895561b49b8ad3a26e5c8e3df8a07777817e7a6ed97f2340",
    ),
    "queries": (
        REFERENCE_ROOT / "queries.parquet",
        "70ded26776bfd56c96501c6033e0e322a6dd11ed296c3309ad89bd9deec84cf9",
    ),
    "guard": (
        REFERENCE_ROOT / "guard_reference.parquet",
        "fee3880a9d3b485abdcca2417952a19baaf70cf35d5dc60fb882378b10f42cca",
    ),
    "query_evidence": (
        REFERENCE_ROOT / "query_evidence.parquet",
        "3ec693b0258b1b1988be226a9aa803656de20e0fcd8aec7feaa960c4fa13e4a8",
    ),
    "ranker": (
        REFERENCE_ROOT / "ranker_reference.parquet",
        "418c8cffec21f030f08baa59e292240e7f4bffbbdc2dcb79b50e83052db48df7",
    ),
    "scenes": (
        REFERENCE_ROOT / "scenes_reference.parquet",
        "9bc4a5f5528f5f4a04126ad3078bcb950e9538b85907dfb4fedd8bd32a8e660c",
    ),
    "partition_manifest": (
        PARTITION_ROOT / "manifest.json",
        "f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33",
    ),
    "partition_assignments": (
        PARTITION_ROOT / "partition_assignments.parquet",
        "f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9",
    ),
    "adjudication_manifest": (
        ADJUDICATION_ROOT / "manifest.json",
        "634ad13c1c2eda0abd7c2921e94ebc1631c070cae8cb3b480514bbfba59e3a8c",
    ),
    "adjudicated_query_ids": (
        ADJUDICATION_ROOT / "current_labels.parquet",
        "e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2",
    ),
}

QUERY_COLUMNS = (
    "query_id",
    "crm_record_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
)
GUARD_COLUMNS = (
    "query_id",
    "predicted_siret",
    "predicted_siren",
    "decision_v412",
    "review_reason_v412",
)
EVIDENCE_COLUMNS = (
    "query_id",
    "direct_candidate_count",
    "direct_siren_count",
    "cross_siren_direct_collision",
    "same_siren_direct_multisite",
)
PARTITION_COLUMNS = ("query_id", "component_id", "partition", "role")
RANKER_COLUMNS = (
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "ranker_rank",
)
OUTPUT_LAYOUT = (
    ("identity", "identity_discovery.parquet"),
    ("identity", "collection_plan.parquet"),
    ("comparison", "docket.parquet"),
    ("comparison", "candidate_context.parquet"),
    ("", "summary.json"),
    ("", "manifest.json"),
)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "mode": int(stat.S_IMODE(value.st_mode)),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "nlink": int(value.st_nlink),
    }


def _directory_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "mode": int(stat.S_IMODE(value.st_mode)),
    }


def _absolute_without_resolve(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _assert_no_symlink_components(path: Path, *, terminal_may_be_missing: bool) -> None:
    absolute = _absolute_without_resolve(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current = current / part
        terminal = index == len(parts) - 1
        try:
            status = current.lstat()
        except FileNotFoundError:
            if terminal and terminal_may_be_missing:
                return
            raise
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"symlink path component forbidden: {current}")
        if not terminal and not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"non-directory path component: {current}")


def _sha256_fd(descriptor: int, size: int, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(chunk_size, size - offset), offset)
        if not chunk:
            raise ValueError("unexpected EOF while hashing input descriptor")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise ValueError("input grew while hashing")
    return digest.hexdigest()


def _open_anchored_path(
    path: Path, *, terminal_directory: bool
) -> tuple[int, list[int], list[dict[str, Any]]]:
    absolute = _absolute_without_resolve(path)
    if ".." in absolute.parts:
        raise ValueError(f"dot-dot path component forbidden: {absolute}")
    parts = absolute.parts[1:]
    root_fd = os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    directory_fds = [root_fd]
    chain = [{"component": "/", **_directory_identity(os.fstat(root_fd))}]
    try:
        for component in parts[:-1]:
            component_status = os.stat(
                component,
                dir_fd=directory_fds[-1],
                follow_symlinks=False,
            )
            if stat.S_ISLNK(component_status.st_mode):
                raise ValueError(f"symlink path component forbidden: {component}")
            descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                dir_fd=directory_fds[-1],
            )
            directory_fds.append(descriptor)
            chain.append(
                {
                    "component": component,
                    **_directory_identity(os.fstat(descriptor)),
                }
            )
        terminal_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        if terminal_directory:
            terminal_flags |= os.O_DIRECTORY
        terminal_status = os.stat(
            parts[-1],
            dir_fd=directory_fds[-1],
            follow_symlinks=False,
        )
        if stat.S_ISLNK(terminal_status.st_mode):
            raise ValueError(f"symlink path component forbidden: {parts[-1]}")
        terminal_fd = os.open(
            parts[-1],
            terminal_flags,
            dir_fd=directory_fds[-1],
        )
        if terminal_directory:
            chain.append(
                {
                    "component": parts[-1],
                    **_directory_identity(os.fstat(terminal_fd)),
                }
            )
        return terminal_fd, directory_fds, chain
    except Exception:
        for descriptor in reversed(directory_fds):
            os.close(descriptor)
        raise


def _reopen_chain(path: Path, *, terminal_directory: bool) -> list[dict[str, Any]]:
    terminal_fd, directory_fds, chain = _open_anchored_path(
        path, terminal_directory=terminal_directory
    )
    try:
        return chain
    finally:
        os.close(terminal_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def _open_copy_input(
    source: Path,
    destination_dir_fd: int,
    destination_name: str,
) -> tuple[int, int, dict[str, Any]]:
    source = _absolute_without_resolve(source)
    descriptor, directory_fds, path_chain = _open_anchored_path(
        source, terminal_directory=False
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"non-regular input forbidden: {source}")
        if before.st_nlink != 1:
            raise ValueError(f"multiply-linked input forbidden: {source}")
        destination_fd = os.open(
            destination_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_dir_fd,
        )
        digest = hashlib.sha256()
        offset = 0
        try:
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1024 * 1024, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise ValueError("unexpected EOF while copying input")
                digest.update(chunk)
                written = 0
                while written < len(chunk):
                    written += os.write(destination_fd, chunk[written:])
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise ValueError("input grew while copying")
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        after = os.fstat(descriptor)
        if _stat_record(after) != _stat_record(before):
            raise ValueError(f"input changed while copying: {source}")
        path_status = source.lstat()
        if _stat_record(path_status) != _stat_record(before):
            raise ValueError(f"input path changed while copying: {source}")
        snapshot = {
            **_stat_record(before),
            "sha256": digest.hexdigest(),
            "path_chain": path_chain,
            "_directory_fds": directory_fds,
        }
        private_fd = os.open(
            destination_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=destination_dir_fd,
        )
        private_status = os.fstat(private_fd)
        if (
            not stat.S_ISREG(private_status.st_mode)
            or private_status.st_nlink != 1
            or private_status.st_size != before.st_size
            or _sha256_fd(private_fd, private_status.st_size) != digest.hexdigest()
        ):
            os.close(private_fd)
            raise ValueError("private input copy failed integrity verification")
        return descriptor, private_fd, snapshot
    except Exception:
        os.close(descriptor)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
        raise


def _revalidate_open_input(
    source: Path,
    descriptor: int,
    snapshot: Mapping[str, Any],
) -> None:
    descriptor_status = os.fstat(descriptor)
    path_status = _absolute_without_resolve(source).lstat()
    expected_stat = {
        key: int(snapshot[key]) for key in _stat_record(descriptor_status)
    }
    if _stat_record(descriptor_status) != expected_stat:
        raise ValueError(f"open input descriptor changed: {source}")
    if _stat_record(path_status) != expected_stat:
        raise ValueError(f"input path was substituted: {source}")
    observed_hash = _sha256_fd(
        descriptor, int(snapshot["size_bytes"])
    )
    if observed_hash != str(snapshot["sha256"]):
        raise ValueError(f"open input bytes changed: {source}")
    if _reopen_chain(source, terminal_directory=False) != snapshot["path_chain"]:
        raise ValueError(f"input ancestor chain was substituted: {source}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _ordered_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_bytes_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_output_root(output_root: Path, *, enforce_canonical: bool) -> Path:
    output_root = _absolute_without_resolve(output_root)
    if enforce_canonical and output_root != DEFAULT_OUTPUT_ROOT:
        raise ValueError("canonical build requires the preregistered output root")
    _assert_no_symlink_components(
        output_root.parent, terminal_may_be_missing=False
    )
    _assert_no_symlink_components(output_root, terminal_may_be_missing=True)
    if not output_root.exists():
        try:
            os.mkdir(output_root, mode=0o700)
            _fsync_directory(output_root.parent)
        except FileExistsError:
            pass
    if output_root.exists():
        status = output_root.lstat()
        if not stat.S_ISDIR(status.st_mode):
            raise ValueError("output root is not a directory")
        if status.st_uid != os.geteuid():
            raise ValueError("output root has the wrong owner")
        if stat.S_IMODE(status.st_mode) != 0o700:
            raise ValueError("pre-existing output root must have mode 0700")
    else:
        raise ValueError("output root could not be created")
    return output_root


def _mkdir_unique_at(directory_fd: int, prefix: str) -> str:
    for _ in range(100):
        name = f"{prefix}{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=directory_fd)
            return name
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique private directory")


def _rename_noreplace_at(
    source_dir_fd: int,
    source_name: str,
    destination_dir_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename without ever replacing an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            encoded_source,
            destination_dir_fd,
            encoded_destination,
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            encoded_source,
            destination_dir_fd,
            encoded_destination,
            0x00000001,  # RENAME_NOREPLACE
        )
    else:
        raise NotImplementedError(
            "atomic no-replace quarantine rename is unavailable"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _quarantine_promoted_target(
    output_fd: int,
    build_id: str,
    staging_fd: int,
) -> str:
    for _ in range(100):
        quarantine_name = f".failed-{build_id}-{secrets.token_hex(8)}"
        try:
            _rename_noreplace_at(
                output_fd,
                build_id,
                output_fd,
                quarantine_name,
            )
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("could not allocate a unique quarantine name")
    os.fsync(output_fd)
    quarantine_fd = _open_directory_at(output_fd, quarantine_name)
    try:
        if _directory_identity(os.fstat(quarantine_fd)) != _directory_identity(
            os.fstat(staging_fd)
        ):
            raise ValueError(
                "quarantined target differs from the promoted staging identity"
            )
    finally:
        os.close(quarantine_fd)
    return quarantine_name


def _open_publication_lock(directory_fd: int) -> int:
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    for _ in range(10):
        try:
            descriptor = os.open(
                ".publication.lock",
                flags | os.O_CREAT,
                0o600,
                dir_fd=directory_fd,
            )
            break
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    ".publication.lock",
                    flags,
                    dir_fd=directory_fd,
                )
                break
            except FileNotFoundError:
                continue
    if descriptor is None:
        raise FileNotFoundError("publication lock could not be opened")
    status = os.fstat(descriptor)
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise ValueError("publication lock identity or permissions are invalid")
    return descriptor


def _open_directory_at(parent_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )


def _file_record_at(directory_fd: int, directory: str, name: str) -> dict[str, Any]:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise ValueError(f"invalid output file: {directory}/{name}")
        relative = f"{directory}/{name}" if directory else name
        return {
            "relative_path": relative,
            "size_bytes": int(status.st_size),
            "sha256": _sha256_fd(descriptor, int(status.st_size)),
        }
    finally:
        os.close(descriptor)


def _tree_records_fd(
    root_fd: int,
    identity_fd: int,
    comparison_fd: int,
) -> list[dict[str, Any]]:
    expected_root = {"identity", "comparison", "summary.json", "manifest.json"}
    observed_root = set(os.listdir(root_fd))
    if "seal.json" in observed_root:
        observed_root.remove("seal.json")
    if observed_root != expected_root:
        raise ValueError(f"unexpected output root entries: {sorted(observed_root)}")
    if set(os.listdir(identity_fd)) != {
        "identity_discovery.parquet",
        "collection_plan.parquet",
    }:
        raise ValueError("unexpected identity output entries")
    if set(os.listdir(comparison_fd)) != {
        "docket.parquet",
        "candidate_context.parquet",
    }:
        raise ValueError("unexpected comparison output entries")
    descriptors = {"": root_fd, "identity": identity_fd, "comparison": comparison_fd}
    return [
        _file_record_at(descriptors[directory], directory, name)
        for directory, name in OUTPUT_LAYOUT
    ]


def _seal_tree_fd(
    root_fd: int,
    identity_fd: int,
    comparison_fd: int,
) -> dict[str, Any]:
    records = _tree_records_fd(root_fd, identity_fd, comparison_fd)
    seal = {
        "schema_version": "sireto-v4.12-r30-docket-seal-1",
        "tree_records": records,
        "tree_sha256": hashlib.sha256(_canonical_json_bytes(records)).hexdigest(),
    }
    _write_bytes_at(root_fd, "seal.json", _canonical_json_bytes(seal) + b"\n")
    for directory, name in (*OUTPUT_LAYOUT, ("", "seal.json")):
        parent_fd = (
            identity_fd
            if directory == "identity"
            else comparison_fd
            if directory == "comparison"
            else root_fd
        )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fchmod(identity_fd, 0o555)
    os.fsync(identity_fd)
    os.fchmod(comparison_fd, 0o555)
    os.fsync(comparison_fd)
    os.fchmod(root_fd, 0o555)
    os.fsync(root_fd)
    return seal


def _verify_sealed_tree_fd(root_fd: int, expected_seal: Mapping[str, Any]) -> None:
    identity_fd = _open_directory_at(root_fd, "identity")
    comparison_fd = _open_directory_at(root_fd, "comparison")
    try:
        if set(os.listdir(root_fd)) != {
            "identity",
            "comparison",
            "summary.json",
            "manifest.json",
            "seal.json",
        }:
            raise ValueError("published root has unexpected entries")
        if stat.S_IMODE(os.fstat(root_fd).st_mode) != 0o555:
            raise ValueError("published root is not sealed 0555")
        for label, descriptor in (
            ("identity", identity_fd),
            ("comparison", comparison_fd),
        ):
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o555:
                raise ValueError(f"published directory is not sealed: {label}")
        for directory, name in (*OUTPUT_LAYOUT, ("", "seal.json")):
            parent_fd = (
                identity_fd
                if directory == "identity"
                else comparison_fd
                if directory == "comparison"
                else root_fd
            )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o444:
                    raise ValueError(f"published file is not sealed: {name}")
            finally:
                os.close(descriptor)
        observed = _tree_records_fd(root_fd, identity_fd, comparison_fd)
        if observed != expected_seal["tree_records"]:
            raise ValueError("published tree records differ from the seal")
        observed_tree_sha = hashlib.sha256(
            _canonical_json_bytes(observed)
        ).hexdigest()
        if observed_tree_sha != str(expected_seal["tree_sha256"]):
            raise ValueError("published tree hash differs from the seal")
        seal_fd = os.open(
            "seal.json",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            seal_status = os.fstat(seal_fd)
            seal_payload = json.loads(
                b"".join(
                    os.pread(seal_fd, min(1024 * 1024, seal_status.st_size - offset), offset)
                    for offset in range(0, seal_status.st_size, 1024 * 1024)
                ).decode("utf-8")
            )
        finally:
            os.close(seal_fd)
        if seal_payload != dict(expected_seal):
            raise ValueError("published seal payload differs")
    finally:
        os.close(comparison_fd)
        os.close(identity_fd)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    try:
        root_fd = _open_directory_at(parent_fd, name)
    except FileNotFoundError:
        return
    try:
        os.fchmod(root_fd, 0o700)
        for child in os.listdir(root_fd):
            status = os.stat(child, dir_fd=root_fd, follow_symlinks=False)
            if stat.S_ISDIR(status.st_mode):
                _remove_tree_at(root_fd, child)
            else:
                os.unlink(child, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _read_parquet(descriptor: int, columns: Iterable[str]) -> pd.DataFrame:
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ValueError("private parquet input is not a regular single-link file")
    chunks = []
    for offset in range(0, status.st_size, 1024 * 1024):
        chunks.append(
            os.pread(
                descriptor,
                min(1024 * 1024, status.st_size - offset),
                offset,
            )
        )
    payload = b"".join(chunks)
    if len(payload) != status.st_size:
        raise ValueError("unexpected EOF while reading private parquet input")
    return pd.read_parquet(io.BytesIO(payload), columns=list(columns))


def _require_columns(frame: pd.DataFrame, expected: Iterable[str], label: str) -> None:
    expected_tuple = tuple(expected)
    if tuple(frame.columns) != expected_tuple:
        raise ValueError(
            f"{label} projection mismatch: {tuple(frame.columns)!r} "
            f"!= {expected_tuple!r}"
        )


def _require_unique_nonempty(frame: pd.DataFrame, column: str, label: str) -> None:
    if not frame[column].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{label} requires string {column} values")
    values = frame[column].astype("string")
    if values.isna().any() or values.str.strip().eq("").any():
        raise ValueError(f"{label} contains an empty {column}")
    if values.duplicated().any():
        raise ValueError(f"{label} contains duplicate {column}")


def _strict_positive_integer_series(
    frame: pd.DataFrame, column: str, label: str
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite {column}")
    numeric = values.to_numpy(dtype=float)
    if (numeric < 1).any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} requires positive integer {column}")
    return values.astype("int64")


def _selection_digest(query_id: str) -> str:
    return hashlib.sha256(
        (SELECTION_DOMAIN + str(query_id)).encode("utf-8")
    ).hexdigest()


def _literal(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return json.dumps(text, ensure_ascii=False)


def _collection_queries(row: Mapping[str, Any]) -> tuple[str, str, str]:
    name = _literal(row["crm_name"])
    postcode = _literal(row["crm_postcode"])
    city = _literal(row["crm_city"])
    address = _literal(row["crm_address"])
    return (
        f"{name} {postcode}",
        f"{name} {city} {address}",
        f"{name} {city} (SIRET OR établissement)",
    )


def _blocked_query_ids(partitions: pd.DataFrame) -> set[str]:
    _require_columns(partitions, PARTITION_COLUMNS, "partition assignments")
    if partitions["query_id"].isna().any():
        raise ValueError("partition assignments contain a null query_id")
    component = partitions["component_id"].astype("string")
    seed = partitions[
        partitions["partition"].astype(str).isin(FORBIDDEN_PARTITIONS)
        | partitions["role"].astype(str).isin(FORBIDDEN_ROLES)
    ]
    forbidden_components = set(
        seed["component_id"].dropna().astype(str).loc[lambda x: x.str.len() > 0]
    )
    blocked = set(seed["query_id"].astype(str))
    if forbidden_components:
        linked = partitions[
            component.notna() & component.astype(str).isin(forbidden_components)
        ]
        blocked.update(linked["query_id"].astype(str))
    return blocked


def _assign_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["stratum"] = "OTHER_REVIEW"
    for column in (
        "cross_siren_direct_collision",
        "same_siren_direct_multisite",
    ):
        if not result[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"{column} must contain strict booleans")
    cross = result["cross_siren_direct_collision"].astype(bool)
    same = result["same_siren_direct_multisite"].astype(bool)
    if (cross & same).any():
        raise ValueError("direct-evidence strata overlap")
    result.loc[cross, "stratum"] = "CROSS_SIREN_COLLISION"
    result.loc[same, "stratum"] = "SAME_SIREN_MULTISITE"
    return result


def _select_docket(
    frame: pd.DataFrame,
    *,
    per_stratum: int,
    expected_selection_sha256: str | None,
) -> tuple[pd.DataFrame, str]:
    selected_parts: list[pd.DataFrame] = []
    for stratum in STRATUM_ORDER:
        part = frame[frame["stratum"].astype(str).eq(stratum)].copy()
        part["selection_digest"] = part["query_id"].astype(str).map(
            _selection_digest
        )
        part = part.sort_values(
            ["selection_digest", "query_id"], kind="mergesort"
        ).head(per_stratum)
        if len(part) != per_stratum:
            raise ValueError(f"not enough eligible rows for {stratum}")
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selection_ordinal"] = range(1, len(selected) + 1)
    ordered_ids = selected["query_id"].astype(str).tolist()
    selection_sha256 = hashlib.sha256(_ordered_json_bytes(ordered_ids)).hexdigest()
    if (
        expected_selection_sha256 is not None
        and selection_sha256 != expected_selection_sha256
    ):
        raise ValueError(
            "selection hash mismatch: "
            f"{selection_sha256} != {expected_selection_sha256}"
        )
    return selected, selection_sha256


def _write_parquet_at(directory_fd: int, name: str, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            pq.write_table(
                table,
                handle,
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=True,
                version="2.6",
                data_page_version="1.0",
                row_group_size=65_536,
            )
            handle.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_frames(
    *,
    queries: pd.DataFrame,
    guard: pd.DataFrame,
    query_evidence: pd.DataFrame,
    partitions: pd.DataFrame,
    adjudicated_query_ids: pd.DataFrame,
    ranker: pd.DataFrame,
    per_stratum: int = 10,
    enforce_canonical_counts: bool = True,
    expected_selection_sha256: str | None = EXPECTED_SELECTION_SHA256,
) -> dict[str, Any]:
    """Build all label-free pilot frames from already projected inputs."""
    _require_columns(queries, QUERY_COLUMNS, "queries")
    _require_columns(guard, GUARD_COLUMNS, "guard")
    _require_columns(query_evidence, EVIDENCE_COLUMNS, "query evidence")
    _require_columns(partitions, PARTITION_COLUMNS, "partition assignments")
    _require_columns(adjudicated_query_ids, ("query_id",), "old adjudications")
    _require_columns(ranker, RANKER_COLUMNS, "ranker")
    for label, frame in (
        ("queries", queries),
        ("guard", guard),
        ("query evidence", query_evidence),
    ):
        _require_unique_nonempty(frame, "query_id", label)
    _require_unique_nonempty(adjudicated_query_ids, "query_id", "old adjudications")

    query_ids = set(queries["query_id"].astype(str))
    if set(guard["query_id"].astype(str)) != query_ids:
        raise ValueError("guard/query population mismatch")
    if set(query_evidence["query_id"].astype(str)) != query_ids:
        raise ValueError("query-evidence/query population mismatch")

    decisions = guard["decision_v412"].astype(str).value_counts().to_dict()
    if enforce_canonical_counts:
        if len(queries) != EXPECTED_COUNTS["query_count"]:
            raise ValueError("canonical query count mismatch")
        if decisions != {
            "AUTO_MATCH": EXPECTED_COUNTS["auto_match_count"],
            "REVIEW": EXPECTED_COUNTS["review_count"],
        }:
            raise ValueError(f"canonical decision counts mismatch: {decisions}")

    merged = queries.merge(guard, on="query_id", validate="one_to_one").merge(
        query_evidence, on="query_id", validate="one_to_one"
    )
    reviews = merged[merged["decision_v412"].astype(str).eq("REVIEW")].copy()
    reviews = _assign_strata(reviews)
    pre_exclusion_counts = {
        str(key): int(value)
        for key, value in reviews["stratum"].value_counts().to_dict().items()
    }
    if enforce_canonical_counts:
        for stratum in STRATUM_ORDER:
            if pre_exclusion_counts.get(stratum, 0) != EXPECTED_COUNTS[stratum]:
                raise ValueError(
                    f"canonical {stratum} count mismatch: "
                    f"{pre_exclusion_counts.get(stratum, 0)}"
                )

    blocked = _blocked_query_ids(partitions)
    blocked.update(adjudicated_query_ids["query_id"].astype(str))
    eligible = reviews[~reviews["query_id"].astype(str).isin(blocked)].copy()
    if enforce_canonical_counts and len(eligible) != EXPECTED_COUNTS["review_count"]:
        raise ValueError("canonical exclusions unexpectedly removed REVIEW rows")

    selected, selection_sha256 = _select_docket(
        eligible,
        per_stratum=per_stratum,
        expected_selection_sha256=expected_selection_sha256,
    )
    selected_ids = set(selected["query_id"].astype(str))

    candidate_context = ranker[
        ranker["query_id"].astype(str).isin(selected_ids)
    ].copy()
    if candidate_context.empty:
        raise ValueError("selected docket has no candidate context")
    candidate_context["query_id"] = candidate_context["query_id"].astype(str)
    for column in ("candidate_siret", "candidate_siren"):
        if not candidate_context[column].map(
            lambda value: isinstance(value, str)
        ).all():
            raise ValueError(f"candidate context requires string {column}")
    if candidate_context.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("candidate context contains duplicate SIRET")
    if not candidate_context["candidate_siret"].astype(str).str.fullmatch(
        r"[0-9]{14}"
    ).all():
        raise ValueError("candidate context contains an invalid SIRET")
    if not candidate_context["candidate_siren"].astype(str).str.fullmatch(
        r"[0-9]{9}"
    ).all():
        raise ValueError("candidate context contains an invalid SIREN")
    if not candidate_context["candidate_siret"].astype(str).str[:9].eq(
        candidate_context["candidate_siren"].astype(str)
    ).all():
        raise ValueError("candidate SIRET/SIREN prefixes disagree")
    candidate_context["ranker_rank"] = _strict_positive_integer_series(
        candidate_context, "ranker_rank", "candidate context"
    )
    candidate_context["retrieval_rank"] = _strict_positive_integer_series(
        candidate_context, "retrieval_rank", "candidate context"
    )
    sizes = candidate_context.groupby("query_id", sort=False).size()
    if set(sizes.index.astype(str)) != selected_ids:
        raise ValueError("candidate context does not cover every selected query")
    if int(sizes.max()) > 100:
        raise ValueError("candidate context exceeds the absolute cap of 100")
    for query_id, group in candidate_context.groupby("query_id", sort=False):
        ranks = sorted(int(value) for value in group["ranker_rank"])
        if ranks != list(range(1, len(group) + 1)):
            raise ValueError(f"non-contiguous ranker ranks for {query_id}")
    candidate_context = candidate_context.sort_values(
        ["query_id", "ranker_rank", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)

    top1 = candidate_context[candidate_context["ranker_rank"].eq(1)][
        ["query_id", "candidate_siret", "candidate_siren"]
    ].copy()
    _require_unique_nonempty(top1, "query_id", "ranker top1")
    top1 = top1.rename(
        columns={
            "candidate_siret": "ranker_top1_siret",
            "candidate_siren": "ranker_top1_siren",
        }
    )
    selected = selected.merge(top1, on="query_id", validate="one_to_one")
    selected = selected.sort_values("selection_ordinal", kind="mergesort")
    if not selected["predicted_siret"].astype(str).eq(
        selected["ranker_top1_siret"].astype(str)
    ).all():
        raise ValueError("guard predicted_siret differs from frozen ranker top1")
    if not selected["predicted_siren"].astype(str).eq(
        selected["ranker_top1_siren"].astype(str)
    ).all():
        raise ValueError("guard predicted_siren differs from frozen ranker top1")

    docket_columns = [
        "query_id",
        "selection_ordinal",
        "stratum",
        "selection_digest",
        "crm_record_id",
        "crm_name",
        "crm_address",
        "crm_postcode",
        "crm_city",
        "crm_insee",
        "predicted_siret",
        "predicted_siren",
        "review_reason_v412",
        "direct_candidate_count",
        "direct_siren_count",
    ]
    docket = selected[docket_columns].copy().reset_index(drop=True)
    identity_discovery = docket[
        [
            "query_id",
            "stratum",
            "crm_record_id",
            "crm_name",
            "crm_address",
            "crm_postcode",
            "crm_city",
            "crm_insee",
        ]
    ].copy()
    forbidden_identity_tokens = (
        "siret",
        "siren",
        "candidate",
        "rank",
        "score",
        "prediction",
        "top1",
        "label",
        "target",
    )
    bad_identity_columns = [
        column
        for column in identity_discovery.columns
        if any(token in column.lower() for token in forbidden_identity_tokens)
    ]
    if bad_identity_columns:
        raise ValueError(
            f"identity-discovery projection leaks model data: {bad_identity_columns}"
        )

    plan_rows: list[dict[str, Any]] = []
    for row in identity_discovery.to_dict(orient="records"):
        for ordinal, query in enumerate(_collection_queries(row), start=1):
            plan_rows.append(
                {
                    "query_id": str(row["query_id"]),
                    "stratum": str(row["stratum"]),
                    "query_ordinal": ordinal,
                    "search_query": query,
                    "max_results_logged": 5,
                    "max_admissible_pages_opened": 2,
                    "max_admissible_pages_total_for_dossier": 6,
                }
            )
    collection_plan = pd.DataFrame(plan_rows).sort_values(
        ["query_id", "query_ordinal"], kind="mergesort"
    )
    if len(collection_plan) != len(docket) * 3:
        raise ValueError("collection plan does not contain three queries per dossier")

    return {
        "docket": docket,
        "identity_discovery": identity_discovery,
        "candidate_context": candidate_context,
        "collection_plan": collection_plan.reset_index(drop=True),
        "selection_sha256": selection_sha256,
        "pre_exclusion_counts": pre_exclusion_counts,
        "blocked_query_id_count": len(blocked),
    }


def build_artifact(
    *,
    input_paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    contract_path: Path,
    output_root: Path,
    enforce_canonical: bool = True,
    per_stratum: int = 10,
) -> Path:
    """Verify inputs, build the label-free artifact, and publish atomically."""
    required = set(CANONICAL_INPUTS)
    if set(input_paths) != required or set(input_hashes) != required:
        raise ValueError("input path/hash keys must match the closed input set")
    output_root = _prepare_output_root(
        output_root, enforce_canonical=enforce_canonical
    )
    output_fd, output_directory_fds, output_chain = _open_anchored_path(
        output_root, terminal_directory=True
    )
    output_parent_snapshot = _stat_record(os.fstat(output_directory_fds[-1]))
    publication_lock_fd = _open_publication_lock(output_fd)
    fcntl.flock(publication_lock_fd, fcntl.LOCK_EX)
    work_name = _mkdir_unique_at(output_fd, ".r30-inputs-")
    work_fd = _open_directory_at(output_fd, work_name)
    open_inputs: dict[str, int] = {}
    private_input_fds: dict[str, int] = {}
    all_opened: list[tuple[Path, int, dict[str, Any]]] = []
    staging_name: str | None = None
    staging_fd: int | None = None
    identity_fd: int | None = None
    comparison_fd: int | None = None
    published_build_id: str | None = None
    try:
        contract_path = _absolute_without_resolve(contract_path)
        builder_path = _absolute_without_resolve(Path(__file__))
        test_path = _absolute_without_resolve(TEST_SOURCE)
        if enforce_canonical and contract_path != CANONICAL_CONTRACT_PATH:
            raise ValueError("canonical build requires the canonical contract path")
        contract_fd, contract_private_fd, contract_snapshot = _open_copy_input(
            contract_path, work_fd, "contract.md"
        )
        private_input_fds["contract"] = contract_private_fd
        all_opened.append((contract_path, contract_fd, contract_snapshot))
        builder_fd, builder_private_fd, builder_snapshot = _open_copy_input(
            builder_path, work_fd, "builder.py"
        )
        private_input_fds["builder"] = builder_private_fd
        all_opened.append((builder_path, builder_fd, builder_snapshot))
        test_fd, test_private_fd, test_snapshot = _open_copy_input(
            test_path, work_fd, "builder_tests.py"
        )
        private_input_fds["tests"] = test_private_fd
        all_opened.append((test_path, test_fd, test_snapshot))
        contract_hash = str(contract_snapshot["sha256"])
        if enforce_canonical and contract_hash != CONTRACT_SHA256:
            raise ValueError("contract hash mismatch")

        observed_hashes: dict[str, str] = {}
        input_snapshots: dict[str, dict[str, Any]] = {}
        for key in sorted(required):
            source = _absolute_without_resolve(input_paths[key])
            if enforce_canonical:
                canonical_path, canonical_hash = CANONICAL_INPUTS[key]
                if source != canonical_path:
                    raise ValueError(f"{key} differs from the canonical path")
                if str(input_hashes[key]) != canonical_hash:
                    raise ValueError(f"{key} differs from the canonical hash")
            destination_name = f"{key}{source.suffix or '.bin'}"
            descriptor, private_fd, snapshot = _open_copy_input(
                source, work_fd, destination_name
            )
            open_inputs[key] = descriptor
            private_input_fds[key] = private_fd
            all_opened.append((source, descriptor, snapshot))
            observed = str(snapshot["sha256"])
            if observed != str(input_hashes[key]):
                raise ValueError(f"{key} hash mismatch")
            observed_hashes[key] = observed
            input_snapshots[key] = {
                item_key: item_value
                for item_key, item_value in snapshot.items()
                if not item_key.startswith("_")
            }
        os.fsync(work_fd)

        projections = {
            "queries": list(QUERY_COLUMNS),
            "guard": list(GUARD_COLUMNS),
            "query_evidence": list(EVIDENCE_COLUMNS),
            "partition_assignments": list(PARTITION_COLUMNS),
            "adjudicated_query_ids": ["query_id"],
            "ranker": list(RANKER_COLUMNS),
        }
        frames = build_frames(
            queries=_read_parquet(private_input_fds["queries"], QUERY_COLUMNS),
            guard=_read_parquet(private_input_fds["guard"], GUARD_COLUMNS),
            query_evidence=_read_parquet(
                private_input_fds["query_evidence"], EVIDENCE_COLUMNS
            ),
            partitions=_read_parquet(
                private_input_fds["partition_assignments"], PARTITION_COLUMNS
            ),
            adjudicated_query_ids=_read_parquet(
                private_input_fds["adjudicated_query_ids"], ("query_id",)
            ),
            ranker=_read_parquet(private_input_fds["ranker"], RANKER_COLUMNS),
            per_stratum=per_stratum,
            enforce_canonical_counts=enforce_canonical,
            expected_selection_sha256=(
                EXPECTED_SELECTION_SHA256 if enforce_canonical else None
            ),
        )
        for source, descriptor, snapshot in all_opened:
            _revalidate_open_input(source, descriptor, snapshot)

        identity = {
            "schema_version": SCHEMA_VERSION,
            "scope": "DOCKET_BUILDER_ONLY_NO_COLLECTION",
            "contract_sha256": contract_hash,
            "builder_sha256": str(builder_snapshot["sha256"]),
            "tests_sha256": str(test_snapshot["sha256"]),
            "inputs": observed_hashes,
            "future_collection_sirene_snapshot": FUTURE_SIRENE_SNAPSHOT,
            "selection_sha256": frames["selection_sha256"],
        }
        build_id = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()[:16]
        target = output_root / build_id
        staging_name = _mkdir_unique_at(output_fd, f".{build_id}.tmp-")
        staging_fd = _open_directory_at(output_fd, staging_name)
        os.mkdir("identity", mode=0o700, dir_fd=staging_fd)
        os.mkdir("comparison", mode=0o700, dir_fd=staging_fd)
        identity_fd = _open_directory_at(staging_fd, "identity")
        comparison_fd = _open_directory_at(staging_fd, "comparison")
        for directory_fd, outputs in (
            (
                identity_fd,
                ("identity_discovery", "collection_plan"),
            ),
            (
                comparison_fd,
                ("docket", "candidate_context"),
            ),
        ):
            for name in outputs:
                _write_parquet_at(directory_fd, f"{name}.parquet", frames[name])
            os.fsync(directory_fd)

        summary = {
            "scope": "DOCKET_BUILDER_ONLY_NO_COLLECTION",
            "query_count": int(len(frames["docket"])),
            "stratum_counts": {
                str(key): int(value)
                for key, value in frames["docket"]["stratum"]
                .value_counts()
                .to_dict()
                .items()
            },
            "candidate_count": int(len(frames["candidate_context"])),
            "candidate_max_per_query": int(
                frames["candidate_context"].groupby("query_id").size().max()
            ),
            "collection_query_count": int(len(frames["collection_plan"])),
            "selection_sha256": frames["selection_sha256"],
            "label_payload_hashed": True,
            "label_columns_deserialized": [],
            "label_semantics_opened": False,
            "public_adjudication_evidence_opened": False,
            "network_access_performed": False,
            "model_training_performed": False,
            "forbidden_population_opened": False,
            "opened_input_keys": sorted(required),
            "deserialized_projections": projections,
        }
        _write_bytes_at(
            staging_fd,
            "summary.json",
            _canonical_json_bytes(summary) + b"\n",
        )
        pre_manifest_records = [
            record
            for directory, name in OUTPUT_LAYOUT
            if name != "manifest.json"
            for record in (
                _file_record_at(
                    identity_fd
                    if directory == "identity"
                    else comparison_fd
                    if directory == "comparison"
                    else staging_fd,
                    directory,
                    name,
                ),
            )
        ]
        outputs = {
            record["relative_path"]: {
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            for record in pre_manifest_records
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_paths": {
                key: str(_absolute_without_resolve(input_paths[key]))
                for key in sorted(required)
            },
            "contract_path": str(contract_path),
            "builder_path": str(builder_path),
            "builder_sha256": str(builder_snapshot["sha256"]),
            "tests_path": str(test_path),
            "tests_sha256": str(test_snapshot["sha256"]),
            "runtime": {
                "python": ".".join(str(value) for value in sys.version_info[:3]),
                "machine": os.uname().machine,
                "system": os.uname().sysname,
                "release": os.uname().release,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "numpy": np.__version__,
            },
            "input_snapshots": input_snapshots,
            "outputs": outputs,
            "summary": summary,
        }
        _write_bytes_at(
            staging_fd,
            "manifest.json",
            _canonical_json_bytes(manifest) + b"\n",
        )
        os.fsync(staging_fd)
        for source, descriptor, snapshot in all_opened:
            _revalidate_open_input(source, descriptor, snapshot)
        if _reopen_chain(
            output_root, terminal_directory=True
        ) != output_chain:
            raise ValueError("output root ancestor chain was substituted")
        if _directory_identity(os.fstat(output_fd)) != {
            key: int(value)
            for key, value in output_chain[-1].items()
            if key != "component"
        }:
            raise ValueError("output root descriptor changed")
        if _stat_record(os.fstat(output_directory_fds[-1])) != output_parent_snapshot:
            raise ValueError("output root parent changed during build")

        try:
            os.stat(build_id, dir_fd=output_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"immutable pilot docket already exists: {target}"
            )
        seal = _seal_tree_fd(staging_fd, identity_fd, comparison_fd)
        os.rename(
            staging_name,
            build_id,
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        published_build_id = build_id
        staging_name = None
        os.fsync(output_fd)
        published_fd = _open_directory_at(output_fd, build_id)
        try:
            if _directory_identity(os.fstat(published_fd)) != _directory_identity(
                os.fstat(staging_fd)
            ):
                raise ValueError("published target differs from sealed staging")
            _verify_sealed_tree_fd(published_fd, seal)
        finally:
            os.close(published_fd)
        published_build_id = None
        return target
    except Exception:
        if staging_name is not None:
            _remove_tree_at(output_fd, staging_name)
        if published_build_id is not None:
            if staging_fd is None:
                raise ValueError("promoted target has no retained staging identity")
            _quarantine_promoted_target(
                output_fd,
                published_build_id,
                staging_fd,
            )
        raise
    finally:
        for descriptor in private_input_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        for descriptor in set(open_inputs.values()) | {
            item[1] for item in all_opened
        }:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for _, _, snapshot in all_opened:
            for descriptor in reversed(snapshot.get("_directory_fds", [])):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        for descriptor in (comparison_fd, identity_fd, staging_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        try:
            _remove_tree_at(output_fd, work_name)
        except OSError:
            pass
        try:
            os.close(work_fd)
        except OSError:
            pass
        try:
            fcntl.flock(publication_lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(publication_lock_fd)
        try:
            os.close(output_fd)
        except OSError:
            pass
        for descriptor in reversed(output_directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/v4_12_review_adjudication_pilot_contract.md"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = build_artifact(
        input_paths={key: path for key, (path, _) in CANONICAL_INPUTS.items()},
        input_hashes={key: sha for key, (_, sha) in CANONICAL_INPUTS.items()},
        contract_path=args.contract,
        output_root=args.output_root,
        enforce_canonical=True,
        per_stratum=10,
    )
    print(target)


if __name__ == "__main__":
    main()
