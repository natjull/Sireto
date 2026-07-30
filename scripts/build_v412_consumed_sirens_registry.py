#!/usr/bin/env python3
"""Build the sealed V4.12 registry of already-consumed SIREN identities."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq


STOP_INPUT = "STOP_INPUT_DRIFT"
STOP_INTEGRITY = "STOP_CONSUMED_SIRENS_INTEGRITY"
STOP_MISMATCH = "STOP_SIRET_SIREN_MISMATCH"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SIREN = re.compile(r"^[0-9]{9}$")
SIRET = re.compile(r"^[0-9]{14}$")
ALLOWED_ROLES = {
    "INPUT_LINEAGE",
    "GROUND_TRUTH_CURRENT",
    "GROUND_TRUTH_HISTORICAL",
    "GROUND_TRUTH_V3",
    "EVALUATION_ORACLE",
}
FORBIDDEN_IDENTITY_TOKENS = (
    "candidate",
    "predicted",
    "top1",
    "retrieval_output",
    "snapshot_neighbor",
    "snapshot_universe",
    "sole_direct",
    "diagnostic_probe",
    "selected_active",
    "direct_active_sirets",
    "siren_component",
    "rank",
    "score",
    "hit",
    "decision",
)
BUILDER_REPOSITORY_PATH = "scripts/build_v412_consumed_sirens_registry.py"
GIT_BINARY = "/usr/bin/git"


class RegistryStop(RuntimeError):
    """Closed failure carrying the contract stop code."""


def _stop(code: str, message: str) -> RegistryStop:
    return RegistryStop(f"{code}: {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)[:-1]).hexdigest()


def _canonical_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _raw_value_hash(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    return environment


def _git_run(
    root: Path,
    arguments: list[str],
    *,
    check: bool,
    text: bool,
) -> subprocess.CompletedProcess[Any]:
    repository = root.resolve(strict=True)
    return subprocess.run(
        [GIT_BINARY, "--no-replace-objects", *arguments],
        cwd=repository,
        env=_git_environment(),
        check=check,
        capture_output=True,
        text=text,
    )


def _git_commit(root: Path) -> str:
    result = _git_run(
        root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise _stop(STOP_INTEGRITY, "HEAD did not resolve to a full commit OID")
    if _git_object_type(root, commit) != "commit":
        raise _stop(STOP_INTEGRITY, "HEAD object is not a commit")
    return commit


def _git_object_type(root: Path, object_id: str) -> str | None:
    result = _git_run(
        root,
        ["cat-file", "-t", object_id],
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_blob_sha256(root: Path, commit: str, repository_path: str) -> str:
    result = _git_run(
        root,
        ["cat-file", "blob", f"{commit}:{repository_path}"],
        check=True,
        text=False,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = _git_run(
        root,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        text=False,
    )
    return result.returncode == 0


def _builder_pin_is_valid(
    builder: Mapping[str, Any],
    script_path: Path,
    repository_root: Path,
) -> bool:
    audited_commit = builder.get("git_commit")
    if (
        builder.get("status") != "PINNED"
        or builder.get("source_path") != BUILDER_REPOSITORY_PATH
        or not isinstance(audited_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", audited_commit)
    ):
        return False
    if _git_object_type(repository_root, audited_commit) != "commit":
        return False
    worktree_hash = file_sha256(script_path)
    if builder.get("source_sha256") != worktree_hash:
        return False
    head = _git_commit(repository_root)
    try:
        blob_hash = _git_blob_sha256(
            repository_root, audited_commit, BUILDER_REPOSITORY_PATH
        )
    except subprocess.CalledProcessError:
        return False
    return blob_hash == worktree_hash and _is_ancestor(
        repository_root, audited_commit, head
    )


def _full_fsync(fd: int) -> None:
    os.fsync(fd)
    if sys.platform == "darwin":
        fcntl.fcntl(fd, 51)  # F_FULLFSYNC


def _sync_directory(path: Path, *, full: bool = False) -> None:
    fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(fd)
        if full and sys.platform == "darwin":
            fcntl.fcntl(fd, 51)
    finally:
        os.close(fd)


def _open_anchored_readonly(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise _stop(STOP_INPUT, f"path is not absolute and anchored: {path}")
    directory_fd = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        components = path.parts[1:]
        for component in components[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            components[-1],
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise _stop(STOP_INPUT, f"secure open failed for {path}: {exc}") from exc
    finally:
        os.close(directory_fd)


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _validate_regular_fd(fd: int, path: Path, expected_size: int | None) -> os.stat_result:
    observed = os.fstat(fd)
    if not stat.S_ISREG(observed.st_mode):
        raise _stop(STOP_INPUT, f"not a regular file: {path}")
    if observed.st_nlink != 1:
        raise _stop(STOP_INPUT, f"hardlink count must be one: {path}")
    if expected_size is not None and observed.st_size != expected_size:
        raise _stop(
            STOP_INPUT,
            f"size drift for {path}: expected {expected_size}, got {observed.st_size}",
        )
    return observed


def _validate_pinned_bytes(
    path: Path, expected_hash: str, expected_size: int | None = None
) -> None:
    fd = _open_anchored_readonly(path)
    try:
        before = _validate_regular_fd(fd, path, expected_size)
        first = _hash_fd(fd)
        second = _hash_fd(fd)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_nlink")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise _stop(STOP_INPUT, f"input mutated while hashing: {path}")
        if first != second or first != expected_hash:
            raise _stop(
                STOP_INPUT,
                f"hash drift for {path}: expected {expected_hash}, got {first}/{second}",
            )
    finally:
        os.close(fd)


def _schema_projection(schema: pa.Schema, fields: list[dict[str, Any]]) -> None:
    for expected in fields:
        index = schema.get_field_index(expected["name"])
        if index < 0:
            raise _stop(STOP_INPUT, f"missing projected column {expected['name']}")
        observed = schema.field(index)
        if str(observed.type) != expected["type"] or observed.nullable != expected["nullable"]:
            raise _stop(
                STOP_INPUT,
                f"schema drift for {expected['name']}: "
                f"{observed.type}/{observed.nullable} != "
                f"{expected['type']}/{expected['nullable']}",
            )


def _read_pinned_parquet(source: Mapping[str, Any]) -> pa.Table:
    path = Path(source["path"])
    fd = _open_anchored_readonly(path)
    try:
        before = _validate_regular_fd(fd, path, int(source["size_bytes"]))
        initial_hash = _hash_fd(fd)
        if initial_hash != source["sha256"]:
            raise _stop(STOP_INPUT, f"hash drift for {path}")
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as handle:
            parquet = pq.ParquetFile(handle)
            if parquet.metadata.num_rows != int(source["row_count"]):
                raise _stop(
                    STOP_INPUT,
                    f"row-count drift for {path}: "
                    f"{parquet.metadata.num_rows} != {source['row_count']}",
                )
            _schema_projection(parquet.schema_arrow, source["projection_schema"])
            if source["projection"] != [
                field["name"] for field in source["projection_schema"]
            ]:
                raise _stop(STOP_INTEGRITY, f"projection/schema order differs for {path}")
            table = parquet.read(columns=source["projection"])
        final_hash = _hash_fd(fd)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_nlink")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise _stop(STOP_INPUT, f"input mutated while reading: {path}")
        if final_hash != initial_hash:
            raise _stop(STOP_INPUT, f"input hash changed while reading: {path}")
        return table
    finally:
        os.close(fd)


def _arrow_type(name: str) -> pa.DataType:
    types = {
        "string": pa.string(),
        "uint32": pa.uint32(),
    }
    if name not in types:
        raise _stop(STOP_INTEGRITY, f"unsupported output Arrow type: {name}")
    return types[name]


def _output_schema(spec: Mapping[str, Any]) -> pa.Schema:
    return pa.schema(
        [
            pa.field(field["name"], _arrow_type(field["type"]), field["nullable"])
            for field in spec["schema"]
        ],
        metadata=None,
    )


def _row_allowed(row: Mapping[str, Any], rule: str) -> bool:
    if rule == "population_status==CONSUMED":
        return row.get("population_status") == "CONSUMED"
    if rule == "population_status==UNSEEN_AND_V411_UNSEEN_EXECUTION_MANIFEST_PIN_VALID":
        return row.get("population_status") == "UNSEEN"
    if rule in {
        "NON_NULL_IDENTITY_ONLY",
        "NON_NULL_IDENTITY_ONLY_MATCH_EXACT_REQUIRES_VALID_IDENTITY",
        "EACH_MAPPING_NON_NULL_IDENTITY_ONLY",
    }:
        return True
    raise _stop(STOP_INTEGRITY, f"unsupported required_filter: {rule}")


def _validate_static_spec(plan: Mapping[str, Any]) -> None:
    expected_files = {
        "sources.json",
        "observations.parquet",
        "consumed_sirens.parquet",
        "rejected_values.parquet",
        "manifest.json",
    }
    if set(plan["outputs"]["exact_files"]) != expected_files:
        raise _stop(STOP_INTEGRITY, "output exact-files contract differs")
    expected_schemas = {
        "observations": [
            ("siren", "string", False),
            ("identity_role", "string", False),
            ("consumption_scope", "string", False),
            ("source_id", "string", False),
            ("source_path", "string", False),
            ("source_sha256", "string", False),
            ("source_manifest_sha256", "string", False),
            ("source_record_locator", "string", False),
            ("source_field", "string", False),
            ("label_kind", "string", True),
            ("derivation", "string", False),
            ("observation_key_sha256", "string", False),
        ],
        "consumed_sirens": [
            ("siren", "string", False),
            ("provenance_count", "uint32", False),
            ("identity_roles_json", "string", False),
            ("consumption_scopes_json", "string", False),
            ("source_ids_json", "string", False),
            ("provenance_payload_sha256", "string", False),
        ],
        "rejected_values": [
            ("source_id", "string", False),
            ("source_record_locator", "string", False),
            ("source_field", "string", False),
            ("identity_role", "string", False),
            ("rejection_reason", "string", False),
            ("raw_value_sha256", "string", True),
        ],
    }
    for name, expected in expected_schemas.items():
        observed = [
            (field["name"], field["type"], field["nullable"])
            for field in plan["outputs"][name]["schema"]
        ]
        if observed != expected:
            raise _stop(STOP_INTEGRITY, f"output schema differs for {name}")
    writer = plan["writer"]
    required_writer = {
        "compression": "zstd",
        "compression_level": 9,
        "data_page_version": "1.0",
        "format_version": "2.6",
        "rechunk_one_chunk_per_column": True,
        "row_group_size": 65536,
        "store_schema": True,
        "use_dictionary": False,
        "write_statistics": True,
    }
    drift = {
        key: {"expected": expected, "observed": writer.get(key)}
        for key, expected in required_writer.items()
        if writer.get(key) != expected
    }
    if drift:
        raise _stop(STOP_INTEGRITY, f"writer parameters differ: {drift}")
    permissions = plan["durability"]["permissions"]
    if permissions != {"directories": "0700", "files": "0600", "umask": "0077"}:
        raise _stop(STOP_INTEGRITY, f"private permissions differ: {permissions}")
    if plan["runtime"].get("models_allowed") is not False:
        raise _stop(STOP_INTEGRITY, "models must be forbidden")
    if plan["runtime"].get("network_allowed") is not False:
        raise _stop(STOP_INTEGRITY, "network must be forbidden")
    if plan["runtime"].get("retrieval_outputs_allowed") is not False:
        raise _stop(STOP_INTEGRITY, "retrieval outputs must be forbidden")


def _mapping_values(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    source: Mapping[str, Any],
    rejected: list[dict[str, Any]],
    locator: str,
) -> tuple[str | None, str | None, str | None]:
    siren_field = mapping.get("siren_field")
    siret_field = mapping.get("siret_field")
    raw_siren = row.get(siren_field) if siren_field else None
    raw_siret = row.get(siret_field) if siret_field else None
    text_siren = _canonical_text(raw_siren)
    text_siret = _canonical_text(raw_siret)
    valid_siren = text_siren if SIREN.fullmatch(text_siren) else None
    valid_siret = text_siret if SIRET.fullmatch(text_siret) else None

    def reject(field: str | None, value: Any, reason: str) -> None:
        if field and _canonical_text(value):
            rejected.append(
                {
                    "source_id": source["id"],
                    "source_record_locator": locator,
                    "source_field": field,
                    "identity_role": mapping["identity_role"],
                    "rejection_reason": reason,
                    "raw_value_sha256": _raw_value_hash(value),
                }
            )

    if text_siren and valid_siren is None:
        reject(siren_field, raw_siren, "INVALID_SIREN_FORMAT")
    if text_siret and valid_siret is None:
        reject(siret_field, raw_siret, "INVALID_SIRET_FORMAT")
    if valid_siren and valid_siret and valid_siren != valid_siret[:9]:
        raise _stop(
            STOP_MISMATCH,
            f"SIRET/SIREN mismatch in {source['id']} at {locator}",
        )
    if valid_siren:
        return valid_siren, valid_siret, "DIRECT_SIREN"
    if valid_siret:
        return valid_siret[:9], valid_siret, "SIRET_PREFIX"
    return None, None, None


def _validate_mapping(mapping: Mapping[str, Any]) -> None:
    if mapping.get("identity_role") not in ALLOWED_ROLES:
        raise _stop(STOP_INTEGRITY, f"forbidden identity role: {mapping}")
    description = "|".join(
        str(mapping.get(key, ""))
        for key in ("source_field", "siren_field", "siret_field")
    ).lower()
    if any(token in description for token in FORBIDDEN_IDENTITY_TOKENS):
        raise _stop(STOP_INTEGRITY, f"candidate/prediction identity mapping: {mapping}")


def _extract(
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    lineage_row_numbers: dict[str, set[int]] = {}
    for source in plan["identity_sources"]:
        for mapping in source["identity_mappings"]:
            _validate_mapping(mapping)
        table = _read_pinned_parquet(source)
        source_counts[source["id"]] = table.num_rows
        if source["id"] in {
            "V411_REGISTRY_CONSUMED_INPUT_LINEAGE",
            "V411_REGISTRY_UNSEEN_NOW_CONSUMED_INPUT_LINEAGE",
        }:
            lineage_row_numbers[source["id"]] = {
                int(value) for value in table.column("source_row_number").to_pylist()
            }
        for row in table.to_pylist():
            if not _row_allowed(row, source["required_filter"]):
                raise _stop(
                    STOP_INTEGRITY,
                    f"row violates required_filter in {source['id']}",
                )
            locator_value = row.get(source["record_locator"])
            if locator_value is None or _canonical_text(locator_value) == "":
                raise _stop(
                    STOP_INTEGRITY,
                    f"missing record locator in {source['id']}",
                )
            locator = _canonical_text(locator_value)
            for mapping in source["identity_mappings"]:
                siren, _siret, derivation = _mapping_values(
                    row, mapping, source, rejected, locator
                )
                label_field = mapping.get("label_kind_field") or source.get(
                    "label_kind_field"
                )
                label_kind = (
                    _canonical_text(row.get(label_field)) if label_field else None
                )
                if label_kind == "MATCH_EXACT" and siren is None:
                    raise _stop(
                        STOP_INTEGRITY,
                        f"MATCH_EXACT without valid identity in {source['id']} at {locator}",
                    )
                if siren is None:
                    continue
                for scope in source["consumption_scopes"]:
                    key_payload = [
                        source["sha256"],
                        locator,
                        mapping["source_field"],
                        mapping["identity_role"],
                        siren,
                    ]
                    observation_key = canonical_hash(key_payload)
                    observations.append(
                        {
                            "siren": siren,
                            "identity_role": mapping["identity_role"],
                            "consumption_scope": scope,
                            "source_id": source["id"],
                            "source_path": source["path"],
                            "source_sha256": source["sha256"],
                            "source_manifest_sha256": source["manifest_sha256"],
                            "source_record_locator": locator,
                            "source_field": mapping["source_field"],
                            "label_kind": label_kind or None,
                            "derivation": derivation,
                            "observation_key_sha256": observation_key,
                        }
                    )
    consumed_id = "V411_REGISTRY_CONSUMED_INPUT_LINEAGE"
    unseen_id = "V411_REGISTRY_UNSEEN_NOW_CONSUMED_INPUT_LINEAGE"
    if consumed_id in lineage_row_numbers or unseen_id in lineage_row_numbers:
        if set(lineage_row_numbers) != {consumed_id, unseen_id}:
            raise _stop(STOP_INTEGRITY, "V4.11 lineage closure source is incomplete")
        consumed_rows = lineage_row_numbers[consumed_id]
        unseen_rows = lineage_row_numbers[unseen_id]
        invariants = plan["invariants"]
        if len(consumed_rows) != int(invariants["expected_v411_consumed_rows"]):
            raise _stop(STOP_INTEGRITY, "V4.11 consumed row-number count differs")
        if len(unseen_rows) != int(invariants["expected_v411_unseen_now_consumed_rows"]):
            raise _stop(STOP_INTEGRITY, "V4.11 unseen row-number count differs")
        if consumed_rows & unseen_rows:
            raise _stop(STOP_INTEGRITY, "V4.11 consumed/unseen row overlap")
        expected_total = int(invariants["expected_v411_total_lineage_rows"])
        if consumed_rows | unseen_rows != set(range(1, expected_total + 1)):
            raise _stop(STOP_INTEGRITY, "V4.11 lineage row-number closure differs")

    deduped: dict[str, dict[str, Any]] = {}
    for row in observations:
        key = row["observation_key_sha256"]
        if key in deduped and deduped[key] != row:
            raise _stop(STOP_INTEGRITY, f"observation-key collision: {key}")
        deduped[key] = row
    if len(deduped) < len(observations):
        observations = list(deduped.values())
    observations.sort(key=lambda row: (row["siren"], row["observation_key_sha256"]))
    rejected.sort(
        key=lambda row: (
            row["source_id"],
            row["source_record_locator"],
            row["source_field"],
            row["identity_role"],
            row["rejection_reason"],
        )
    )
    return observations, rejected, source_counts


def _aggregate(observations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["siren"]), []).append(observation)
    output: list[dict[str, Any]] = []
    for siren in sorted(grouped):
        rows = grouped[siren]
        if not SIREN.fullmatch(siren):
            raise _stop(STOP_INTEGRITY, f"invalid final SIREN: {siren}")
        roles = sorted({str(row["identity_role"]) for row in rows})
        scopes = sorted({str(row["consumption_scope"]) for row in rows})
        sources = sorted({str(row["source_id"]) for row in rows})
        keys = sorted(str(row["observation_key_sha256"]) for row in rows)
        output.append(
            {
                "siren": siren,
                "provenance_count": len(rows),
                "identity_roles_json": canonical_bytes(roles)[:-1].decode("utf-8"),
                "consumption_scopes_json": canonical_bytes(scopes)[:-1].decode("utf-8"),
                "source_ids_json": canonical_bytes(sources)[:-1].decode("utf-8"),
                "provenance_payload_sha256": canonical_hash(keys),
            }
        )
    return output


def _table(rows: list[dict[str, Any]], spec: Mapping[str, Any]) -> pa.Table:
    schema = _output_schema(spec)
    return pa.Table.from_pylist(rows, schema=schema).combine_chunks()


def _write_exclusive_bytes_at(parent_fd: int, name: str, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        _full_fsync(fd)
    finally:
        os.close(fd)
    os.chmod(name, 0o600, dir_fd=parent_fd, follow_symlinks=False)


def _write_exclusive_parquet_at(
    parent_fd: int, name: str, table: pa.Table, writer: Mapping[str, Any]
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    try:
        with os.fdopen(os.dup(fd), "wb") as handle:
            pq.write_table(
                table.replace_schema_metadata(None).combine_chunks(),
                handle,
                row_group_size=int(writer["row_group_size"]),
                version=writer["format_version"],
                compression=writer["compression"],
                compression_level=int(writer["compression_level"]),
                use_dictionary=bool(writer["use_dictionary"]),
                write_statistics=bool(writer["write_statistics"]),
                data_page_version=writer["data_page_version"],
                store_schema=bool(writer["store_schema"]),
            )
            handle.flush()
        _full_fsync(fd)
    finally:
        os.close(fd)
    os.chmod(name, 0o600, dir_fd=parent_fd, follow_symlinks=False)


def _rename_exclusive_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.renameatx_np
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    result = function(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        0x00000004,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination_name)
        raise OSError(error, os.strerror(error), destination_name)


def _open_private_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise _stop(STOP_INTEGRITY, f"output root must be absolute: {path}")
    parent_fd = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    failed = True
    try:
        for component in path.parts[1:]:
            created = False
            try:
                child_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=parent_fd)
                created = True
                child_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            if created or component == path.parts[-1]:
                os.fchmod(child_fd, 0o700)
            os.close(parent_fd)
            parent_fd = child_fd
        failed = False
        return parent_fd
    except OSError as exc:
        raise _stop(STOP_INTEGRITY, f"unsafe output directory {path}: {exc}") from exc
    finally:
        if failed:
            os.close(parent_fd)


def _sync_directory_fd(fd: int, *, full: bool = False) -> None:
    os.fsync(fd)
    if full and sys.platform == "darwin":
        fcntl.fcntl(fd, 51)


def _open_child_directory(parent_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )


def _file_info_at(parent_fd: int, name: str) -> dict[str, Any]:
    fd = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise _stop(STOP_INTEGRITY, f"unsafe staged file: {name}")
        return {"size_bytes": observed.st_size, "sha256": _hash_fd(fd)}
    finally:
        os.close(fd)


def _read_bytes_at(parent_fd: int, name: str) -> bytes:
    fd = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise _stop(STOP_INTEGRITY, f"unsafe staged file: {name}")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _parquet_file_at(parent_fd: int, name: str) -> tuple[pq.ParquetFile, Any]:
    fd = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    handle = os.fdopen(fd, "rb")
    try:
        return pq.ParquetFile(handle), handle
    except Exception:
        handle.close()
        raise


def _read_parquet_table_at(
    parent_fd: int, name: str, expected_schema: pa.Schema
) -> pa.Table:
    parquet, handle = _parquet_file_at(parent_fd, name)
    try:
        if parquet.schema_arrow != expected_schema:
            raise _stop(STOP_INTEGRITY, f"recovery schema differs for {name}")
        return parquet.read().combine_chunks()
    finally:
        handle.close()


def _validate_recovery_semantics(
    stage_fd: int,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    observations_table = _read_parquet_table_at(
        stage_fd,
        "observations.parquet",
        _output_schema(plan["outputs"]["observations"]),
    )
    consumed_table = _read_parquet_table_at(
        stage_fd,
        "consumed_sirens.parquet",
        _output_schema(plan["outputs"]["consumed_sirens"]),
    )
    rejected_table = _read_parquet_table_at(
        stage_fd,
        "rejected_values.parquet",
        _output_schema(plan["outputs"]["rejected_values"]),
    )
    observations = observations_table.to_pylist()
    consumed = consumed_table.to_pylist()
    rejected = rejected_table.to_pylist()
    if manifest.get("observation_count") != len(observations):
        raise _stop(STOP_INTEGRITY, "recovery observation count differs")
    if manifest.get("unique_siren_count") != len(consumed):
        raise _stop(STOP_INTEGRITY, "recovery unique-SIREN count differs")
    if manifest.get("observations_logical_sha256") != canonical_hash(observations):
        raise _stop(STOP_INTEGRITY, "recovery observations logical hash differs")
    if manifest.get("sirens_logical_sha256") != canonical_hash(consumed):
        raise _stop(STOP_INTEGRITY, "recovery SIREN logical hash differs")

    source_by_id = {
        source["id"]: source for source in plan["identity_sources"]
    }
    expected_source_counts = {
        source["id"]: int(source["row_count"])
        for source in plan["identity_sources"]
    }
    if manifest.get("source_row_counts") != expected_source_counts:
        raise _stop(STOP_INTEGRITY, "recovery source row counts differ")
    seen_observation_keys: set[str] = set()
    previous_observation_sort: tuple[str, str] | None = None
    for row in observations:
        siren = row["siren"]
        key = row["observation_key_sha256"]
        if not SIREN.fullmatch(siren) or not HEX64.fullmatch(key):
            raise _stop(STOP_INTEGRITY, "invalid recovery observation identity")
        if key in seen_observation_keys:
            raise _stop(STOP_INTEGRITY, "duplicate recovery observation key")
        seen_observation_keys.add(key)
        sort_key = (siren, key)
        if previous_observation_sort is not None and sort_key <= previous_observation_sort:
            raise _stop(STOP_INTEGRITY, "recovery observations are not strictly sorted")
        previous_observation_sort = sort_key
        if row["identity_role"] not in ALLOWED_ROLES:
            raise _stop(STOP_INTEGRITY, "invalid recovery identity role")
        if row["derivation"] not in {"DIRECT_SIREN", "SIRET_PREFIX"}:
            raise _stop(STOP_INTEGRITY, "invalid recovery derivation")
        source = source_by_id.get(row["source_id"])
        if source is None:
            raise _stop(STOP_INTEGRITY, "unknown recovery source id")
        if (
            row["source_path"] != source["path"]
            or row["source_sha256"] != source["sha256"]
            or row["source_manifest_sha256"] != source["manifest_sha256"]
            or row["consumption_scope"] not in source["consumption_scopes"]
            or not row["source_record_locator"]
        ):
            raise _stop(STOP_INTEGRITY, "recovery observation provenance differs")
        declared_mappings = {
            (mapping["source_field"], mapping["identity_role"])
            for mapping in source["identity_mappings"]
        }
        if (row["source_field"], row["identity_role"]) not in declared_mappings:
            raise _stop(STOP_INTEGRITY, "undeclared recovery observation mapping")
        expected_key = canonical_hash(
            [
                row["source_sha256"],
                row["source_record_locator"],
                row["source_field"],
                row["identity_role"],
                siren,
            ]
        )
        if key != expected_key:
            raise _stop(STOP_INTEGRITY, "recovery observation key differs")

    expected_consumed = _aggregate(observations)
    if consumed != expected_consumed:
        raise _stop(
            STOP_INTEGRITY,
            "recovery consumed registry differs from observation aggregation",
        )
    if [row["siren"] for row in consumed] != sorted(
        {row["siren"] for row in consumed}
    ):
        raise _stop(STOP_INTEGRITY, "recovery consumed SIRENs are not unique/sorted")
    for row in consumed:
        if (
            not SIREN.fullmatch(row["siren"])
            or int(row["provenance_count"]) <= 0
            or not HEX64.fullmatch(row["provenance_payload_sha256"])
        ):
            raise _stop(STOP_INTEGRITY, "invalid recovery consumed row")
        for field in (
            "identity_roles_json",
            "consumption_scopes_json",
            "source_ids_json",
        ):
            values = json.loads(row[field])
            if values != sorted(set(values)):
                raise _stop(STOP_INTEGRITY, f"invalid recovery {field}")

    previous_rejected_sort: tuple[str, str, str, str, str] | None = None
    for row in rejected:
        if row["source_id"] not in source_by_id:
            raise _stop(STOP_INTEGRITY, "unknown rejected source id")
        if row["identity_role"] not in ALLOWED_ROLES:
            raise _stop(STOP_INTEGRITY, "invalid rejected identity role")
        raw_hash = row["raw_value_sha256"]
        if raw_hash is not None and not HEX64.fullmatch(raw_hash):
            raise _stop(STOP_INTEGRITY, "invalid rejected raw-value hash")
        sort_key = (
            row["source_id"],
            row["source_record_locator"],
            row["source_field"],
            row["identity_role"],
            row["rejection_reason"],
        )
        if previous_rejected_sort is not None and sort_key < previous_rejected_sort:
            raise _stop(STOP_INTEGRITY, "rejected rows are not sorted")
        previous_rejected_sort = sort_key
    expected_role_counts = {
        role: sum(1 for row in observations if row["identity_role"] == role)
        for role in sorted(ALLOWED_ROLES)
    }
    if manifest.get("identity_role_counts") != expected_role_counts:
        raise _stop(STOP_INTEGRITY, "recovery identity-role counts differ")
    expected_rejection_counts = {
        reason: sum(1 for row in rejected if row["rejection_reason"] == reason)
        for reason in sorted({row["rejection_reason"] for row in rejected})
    }
    if manifest.get("rejection_counts") != expected_rejection_counts:
        raise _stop(STOP_INTEGRITY, "recovery rejection counts differ")


def _validate_complete_stage(
    stage_fd: int,
    plan: Mapping[str, Any],
    *,
    build_id: str,
    plan_hash: str,
    contract_hash: str,
    script_hash: str,
    commit: str,
) -> None:
    expected_names = sorted(plan["outputs"]["exact_files"])
    if sorted(os.listdir(stage_fd)) != expected_names:
        raise _stop(
            STOP_INTEGRITY,
            "recovery tree is partial or contains undeclared files",
        )
    if stat.S_IMODE(os.fstat(stage_fd).st_mode) != 0o700:
        raise _stop(STOP_INTEGRITY, "recovery directory mode differs from 0700")
    manifest_raw = _read_bytes_at(stage_fd, "manifest.json")
    manifest = json.loads(manifest_raw)
    if manifest_raw != canonical_bytes(manifest):
        raise _stop(STOP_INTEGRITY, "recovery manifest is not canonical")
    expected_identity_hashes = {
        source["id"]: source["sha256"] for source in plan["identity_sources"]
    }
    expected_event_hashes = {
        event["event_role"]: event["sha256"]
        for event in plan["event_only_manifests"]
    }
    expected_header = {
        "build_id": build_id,
        "builder_git_commit": commit,
        "builder_source_sha256": script_hash,
        "contract_sha256": contract_hash,
        "plan_sha256": plan_hash,
        "input_source_hashes": expected_identity_hashes,
        "event_manifest_hashes": expected_event_hashes,
    }
    for field, expected in expected_header.items():
        if manifest.get(field) != expected:
            raise _stop(STOP_INTEGRITY, f"recovery manifest differs at {field}")
    sources_expected = canonical_bytes(
        {
            "identity_sources": plan["identity_sources"],
            "event_only_manifests": plan["event_only_manifests"],
        }
    )
    if _read_bytes_at(stage_fd, "sources.json") != sources_expected:
        raise _stop(STOP_INTEGRITY, "recovery sources.json differs")
    files = {
        filename: _file_info_at(stage_fd, filename)
        for filename in (
            "sources.json",
            "observations.parquet",
            "consumed_sirens.parquet",
            "rejected_values.parquet",
        )
    }
    if manifest.get("files") != files:
        raise _stop(STOP_INTEGRITY, "recovery payload hashes differ")
    if manifest.get("tree_payload_sha256") != canonical_hash(files):
        raise _stop(STOP_INTEGRITY, "recovery tree hash differs")
    for filename, output_name in (
        ("observations.parquet", "observations"),
        ("consumed_sirens.parquet", "consumed_sirens"),
        ("rejected_values.parquet", "rejected_values"),
    ):
        parquet, handle = _parquet_file_at(stage_fd, filename)
        try:
            expected_schema = _output_schema(plan["outputs"][output_name])
            if parquet.schema_arrow != expected_schema:
                raise _stop(
                    STOP_INTEGRITY, f"recovery schema differs for {filename}"
                )
            rows = parquet.metadata.num_rows
            expected_groups = max(
                1, math.ceil(rows / int(plan["writer"]["row_group_size"]))
            )
            if parquet.metadata.num_row_groups != expected_groups:
                raise _stop(
                    STOP_INTEGRITY,
                    f"recovery row-group count differs for {filename}",
                )
        finally:
            handle.close()
    _validate_recovery_semantics(stage_fd, plan, manifest)
    for filename in expected_names:
        descriptor = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=stage_fd,
        )
        try:
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                raise _stop(
                    STOP_INTEGRITY, f"recovery file mode differs for {filename}"
                )
        finally:
            os.close(descriptor)


def _recover_complete_stage(
    root_fd: int,
    plan: Mapping[str, Any],
    *,
    build_id: str,
    plan_hash: str,
    contract_hash: str,
    script_hash: str,
    commit: str,
) -> bool:
    prefix = f".tmp-{build_id}-"
    candidates = sorted(
        name for name in os.listdir(root_fd) if name.startswith(prefix)
    )
    if not candidates:
        return False
    if len(candidates) != 1:
        raise _stop(STOP_INTEGRITY, "multiple recovery trees for one build")
    stage_name = candidates[0]
    stage_fd = _open_child_directory(root_fd, stage_name)
    try:
        _validate_complete_stage(
            stage_fd,
            plan,
            build_id=build_id,
            plan_hash=plan_hash,
            contract_hash=contract_hash,
            script_hash=script_hash,
            commit=commit,
        )
    finally:
        os.close(stage_fd)
    _rename_exclusive_at(root_fd, stage_name, root_fd, build_id)
    _sync_directory_fd(root_fd, full=True)
    return True


def _validate_runtime(plan: Mapping[str, Any]) -> None:
    runtime = plan["runtime"]
    observed = {
        "os": "macOS" if sys.platform == "darwin" else sys.platform,
        "architecture": os.uname().machine,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "pyarrow": pa.__version__,
    }
    expected = {key: runtime[key] for key in observed}
    if observed != expected:
        raise _stop(STOP_INTEGRITY, f"runtime drift: {observed} != {expected}")
    if runtime.get("pandas_serialization_allowed") is not False:
        raise _stop(STOP_INTEGRITY, "pandas serialization must be forbidden")


def validate_plan(
    plan_path: Path,
    contract_path: Path,
    script_path: Path,
    *,
    require_builder_pin: bool,
) -> tuple[dict[str, Any], str, str, str]:
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    if plan_raw != canonical_bytes(plan):
        raise _stop(STOP_INTEGRITY, "plan JSON is not canonical")
    plan_hash = hashlib.sha256(plan_raw).hexdigest()
    contract_hash = file_sha256(contract_path)
    if plan["contract"]["path"] != str(contract_path):
        raise _stop(STOP_INTEGRITY, "contract path differs from plan")
    if plan["contract"]["sha256"] != contract_hash:
        raise _stop(STOP_INTEGRITY, "contract hash differs from plan")
    if plan["invariants"]["expected_identity_source_count"] != len(
        plan["identity_sources"]
    ):
        raise _stop(STOP_INTEGRITY, "identity source count differs")
    if plan["invariants"]["expected_event_manifest_count"] != len(
        plan["event_only_manifests"]
    ):
        raise _stop(STOP_INTEGRITY, "event manifest count differs")
    if plan["invariants"]["candidate_or_prediction_identity_count"] != 0:
        raise _stop(STOP_INTEGRITY, "candidate identity invariant is nonzero")
    _validate_static_spec(plan)
    for source in plan["identity_sources"]:
        for mapping in source["identity_mappings"]:
            _validate_mapping(mapping)
        _validate_pinned_bytes(
            Path(source["manifest_path"]), source["manifest_sha256"]
        )
    for event in plan["event_only_manifests"]:
        _validate_pinned_bytes(
            Path(event["path"]), event["sha256"], int(event["size_bytes"])
        )
    _validate_runtime(plan)
    script_hash = file_sha256(script_path)
    repository_root = script_path.resolve().parent.parent
    commit = _git_commit(repository_root)
    builder = plan["build"]["builder"]
    pin_valid = _builder_pin_is_valid(builder, script_path, repository_root)
    if require_builder_pin and not pin_valid:
        raise _stop(
            STOP_INTEGRITY,
            "builder worktree/blob/ancestry/path/status pin is invalid",
        )
    audited_commit = builder.get("git_commit") if pin_valid else commit
    return plan, plan_hash, contract_hash, audited_commit


def validate_inputs(plan: Mapping[str, Any]) -> None:
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for source in plan["identity_sources"]:
        if source["id"] in seen_ids or source["path"] in seen_paths:
            raise _stop(STOP_INTEGRITY, "duplicate identity source id/path")
        seen_ids.add(source["id"])
        seen_paths.add(source["path"])
        _read_pinned_parquet(source)


def _build_id(
    plan: Mapping[str, Any],
    plan_hash: str,
    contract_hash: str,
    script_hash: str,
    commit: str,
) -> str:
    projection = {
        "schema_version": plan["schema_version"],
        "normalization_version": plan["normalization"]["version"],
        "identity_sources": [
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "projection": source["projection"],
                "identity_mappings": source["identity_mappings"],
            }
            for source in plan["identity_sources"]
        ],
        "event_only_manifests": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in plan["event_only_manifests"]
        ],
        "contract_sha256": contract_hash,
        "plan_sha256": plan_hash,
        "builder_source_sha256": script_hash,
        "builder_git_commit": commit,
    }
    return canonical_hash(projection)[: int(plan["build"]["build_id_length"])]


def _build_registry_impl(
    plan_path: Path,
    contract_path: Path,
    output_root: Path,
    script_path: Path,
) -> Path:
    plan, plan_hash, contract_hash, commit = validate_plan(
        plan_path, contract_path, script_path, require_builder_pin=True
    )
    script_hash = file_sha256(script_path)
    build_id = _build_id(
        plan, plan_hash, contract_hash, script_hash, commit
    )
    root_fd = _open_private_directory(output_root)
    try:
        if build_id in os.listdir(root_fd):
            raise FileExistsError(output_root / build_id)
        if _recover_complete_stage(
            root_fd,
            plan,
            build_id=build_id,
            plan_hash=plan_hash,
            contract_hash=contract_hash,
            script_hash=script_hash,
            commit=commit,
        ):
            return output_root / build_id

        # The public build API performs the same duplicate/source validation as
        # the CLI. Recovery above deliberately precedes all payload rereads.
        validate_inputs(plan)
        observations, rejected, source_counts = _extract(plan)
        consumed = _aggregate(observations)
        if not observations or not consumed:
            raise _stop(STOP_INTEGRITY, "registry cannot be empty")
        writer = plan["writer"]
        outputs = plan["outputs"]
        tables = {
            "observations.parquet": _table(
                observations, outputs["observations"]
            ),
            "consumed_sirens.parquet": _table(
                consumed, outputs["consumed_sirens"]
            ),
            "rejected_values.parquet": _table(
                rejected, outputs["rejected_values"]
            ),
        }
        stage_name = f".tmp-{build_id}-{os.getpid()}"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=root_fd)
        except FileExistsError as exc:
            raise _stop(
                STOP_INTEGRITY,
                f"temporary build already exists: {stage_name}",
            ) from exc
        stage_fd = _open_child_directory(root_fd, stage_name)
        try:
            os.fchmod(stage_fd, 0o700)
            sources_payload = {
                "identity_sources": plan["identity_sources"],
                "event_only_manifests": plan["event_only_manifests"],
            }
            _write_exclusive_bytes_at(
                stage_fd, "sources.json", canonical_bytes(sources_payload)
            )
            for filename, table in tables.items():
                _write_exclusive_parquet_at(
                    stage_fd, filename, table, writer
                )
            files = {
                filename: _file_info_at(stage_fd, filename)
                for filename in (
                    "sources.json",
                    "observations.parquet",
                    "consumed_sirens.parquet",
                    "rejected_values.parquet",
                )
            }
            manifest = {
                "schema_version": plan["schema_version"],
                "build_id": build_id,
                "builder_git_commit": commit,
                "builder_source_sha256": script_hash,
                "contract_sha256": contract_hash,
                "plan_sha256": plan_hash,
                "input_source_hashes": {
                    source["id"]: source["sha256"]
                    for source in plan["identity_sources"]
                },
                "event_manifest_hashes": {
                    event["event_role"]: event["sha256"]
                    for event in plan["event_only_manifests"]
                },
                "source_row_counts": source_counts,
                "identity_role_counts": {
                    role: sum(
                        1
                        for row in observations
                        if row["identity_role"] == role
                    )
                    for role in sorted(ALLOWED_ROLES)
                },
                "rejection_counts": {
                    reason: sum(
                        1
                        for row in rejected
                        if row["rejection_reason"] == reason
                    )
                    for reason in sorted(
                        {row["rejection_reason"] for row in rejected}
                    )
                },
                "observation_count": len(observations),
                "unique_siren_count": len(consumed),
                "observations_logical_sha256": canonical_hash(observations),
                "sirens_logical_sha256": canonical_hash(consumed),
                "files": files,
                "tree_payload_sha256": canonical_hash(files),
            }
            required = set(outputs["manifest"]["required_fields"])
            if set(manifest) != required:
                raise _stop(
                    STOP_INTEGRITY,
                    f"manifest fields differ: {set(manifest) ^ required}",
                )
            _write_exclusive_bytes_at(
                stage_fd, "manifest.json", canonical_bytes(manifest)
            )
            _sync_directory_fd(stage_fd)
            _validate_complete_stage(
                stage_fd,
                plan,
                build_id=build_id,
                plan_hash=plan_hash,
                contract_hash=contract_hash,
                script_hash=script_hash,
                commit=commit,
            )
        finally:
            os.close(stage_fd)
        _rename_exclusive_at(root_fd, stage_name, root_fd, build_id)
        _sync_directory_fd(root_fd, full=True)
        return output_root / build_id
    finally:
        os.close(root_fd)


def build_registry(
    plan_path: Path,
    contract_path: Path,
    output_root: Path,
    script_path: Path,
) -> Path:
    previous_umask = os.umask(0o077)
    try:
        return _build_registry_impl(
            plan_path, contract_path, output_root, script_path
        )
    finally:
        os.umask(previous_umask)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("config/v4_12_consumed_sirens_registry_plan.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/v4_12_consumed_sirens_registry_contract.md"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    script_path = Path(__file__).resolve()
    try:
        plan, plan_hash, contract_hash, _commit = validate_plan(
            args.plan,
            args.contract,
            script_path,
            require_builder_pin=not args.validate_only,
        )
        validate_inputs(plan)
        if args.validate_only:
            builder = plan["build"]["builder"]
            repository_root = script_path.parent.parent
            result = {
                "status": "VALIDATED_NO_BUILD",
                "plan_sha256": plan_hash,
                "contract_sha256": contract_hash,
                "builder_pinned_for_execution": _builder_pin_is_valid(
                    builder, script_path, repository_root
                ),
            }
        else:
            if args.output_root is None:
                raise _stop(STOP_INTEGRITY, "--output-root is required for build")
            destination = build_registry(
                args.plan, args.contract, args.output_root, script_path
            )
            result = {"status": "GO_V412_CONSUMED_SIRENS_REGISTRY", "path": str(destination)}
        sys.stdout.buffer.write(canonical_bytes(result))
        return 0
    except (RegistryStop, FileExistsError, OSError, KeyError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except Exception as exc:  # pragma: no cover - final CLI fail-closed boundary
        sys.stderr.write(
            f"{STOP_INTEGRITY}: unexpected {type(exc).__name__}: {exc}\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
