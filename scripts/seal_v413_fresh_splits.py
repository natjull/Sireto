#!/usr/bin/env python3
"""Deterministically assign V4.13 qualified queries to sealed split manifests.

This module is deliberately model- and retrieval-free.  Its input is a
private, already-qualified row projection containing only the opaque query ID,
an optional source group and every authoritatively known SIREN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DOMAIN = b"SIRETO-V413-FRESH-SPLIT\0"
FIT_UPPER = 12912720851596686131
DEV_UPPER = 15679732462653118873
OPAQUE_RE = re.compile(r"^[a-p]{64}$")
SIREN_RE = re.compile(r"^[0-9]{9}$")
SCHEMA = "sireto-v4.13-split-input-row-1"
MANIFEST_SCHEMA = "sireto-v4.13-split-manifest-1"


class SplitStopped(RuntimeError):
    """Fail-closed split construction."""


def _stop(message: str) -> None:
    raise SplitStopped(f"STOP_V413_SPLIT: {message}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        self.parent[upper] = lower


def _validate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(rows)
    if not materialized:
        _stop("at least one row required")
    expected = {
        "schema_version",
        "query_id",
        "source_group_id",
        "authoritative_sirens",
    }
    seen: set[str] = set()
    for ordinal, row in enumerate(materialized, start=1):
        if not isinstance(row, dict) or set(row) != expected:
            _stop(f"row {ordinal} schema mismatch")
        if row["schema_version"] != SCHEMA:
            _stop(f"row {ordinal} schema_version mismatch")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not OPAQUE_RE.fullmatch(query_id):
            _stop(f"row {ordinal} invalid query_id")
        if query_id in seen:
            _stop(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        group = row["source_group_id"]
        if group is not None and (not isinstance(group, str) or not group):
            _stop(f"row {ordinal} invalid source_group_id")
        sirens = row["authoritative_sirens"]
        if (
            not isinstance(sirens, list)
            or len(sirens) != len(set(sirens))
            or sirens != sorted(sirens)
            or any(not isinstance(siren, str) or not SIREN_RE.fullmatch(siren) for siren in sirens)
        ):
            _stop(f"row {ordinal} authoritative_sirens must be sorted unique SIREN")
    return sorted(materialized, key=lambda row: row["query_id"])


def assign_components(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the complete deterministic query -> component/split assignment."""

    valid = _validate_rows(rows)
    union = UnionFind()
    first_by_group: dict[str, str] = {}
    first_by_siren: dict[str, str] = {}
    for row in valid:
        query_id = row["query_id"]
        union.find(query_id)
        group = row["source_group_id"]
        if group is not None:
            if group in first_by_group:
                union.union(query_id, first_by_group[group])
            else:
                first_by_group[group] = query_id
        for siren in row["authoritative_sirens"]:
            if siren in first_by_siren:
                union.union(query_id, first_by_siren[siren])
            else:
                first_by_siren[siren] = query_id

    members: dict[str, list[str]] = defaultdict(list)
    for row in valid:
        members[union.find(row["query_id"])].append(row["query_id"])

    result: dict[str, dict[str, Any]] = {}
    for component_members in members.values():
        ordered = sorted(component_members)
        component_key = canonical_json_bytes(ordered)
        component_sha256 = hashlib.sha256(component_key).hexdigest()
        digest = hashlib.sha256(DOMAIN + component_key).digest()
        value = int.from_bytes(digest[:8], "big")
        split = "fit" if value < FIT_UPPER else "dev" if value < DEV_UPPER else "test"
        for query_id in ordered:
            result[query_id] = {
                "component_sha256": component_sha256,
                "split": split,
                "split_uint64": value,
            }
    return dict(sorted(result.items()))


def build_manifests(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    assignments = assign_components(rows)
    manifests: dict[str, dict[str, Any]] = {}
    for split in ("fit", "dev", "test"):
        selected = [
            {
                "query_id": query_id,
                "component_sha256": values["component_sha256"],
                "split_uint64": values["split_uint64"],
            }
            for query_id, values in assignments.items()
            if values["split"] == split
        ]
        manifests[split] = {
            "schema_version": MANIFEST_SCHEMA,
            "split": split,
            "query_count": len(selected),
            "component_count": len(
                {item["component_sha256"] for item in selected}
            ),
            "assignments": selected,
        }
    return manifests


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def seal_manifests(rows: Iterable[dict[str, Any]], output_root: Path) -> dict[str, str]:
    manifests = build_manifests(rows)
    resolved = output_root.resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    if resolved == temporary or temporary not in resolved.parents or output_root.is_symlink():
        _stop("synthetic split output must be below OS tmp without symlink")
    if resolved.exists():
        raise FileExistsError(resolved)
    staging = resolved.parent / f".{resolved.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700, parents=False, exist_ok=False)
    hashes: dict[str, str] = {}
    try:
        for split, manifest in manifests.items():
            payload = canonical_json_bytes(manifest)
            path = staging / split / "split_manifest.json"
            _write_exclusive(path, payload)
            hashes[split] = hashlib.sha256(payload).hexdigest()
        staging_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.rename(staging, resolved)
        parent_fd = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    if temporary not in resolved.parents or path.is_symlink():
        _stop("synthetic split input must be below OS tmp without symlink")
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except Exception as exc:
            _stop(f"invalid JSONL row {ordinal}: {exc}")
        if not isinstance(value, dict):
            _stop(f"JSONL row {ordinal} must be object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-qualified-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    hashes = seal_manifests(
        _read_jsonl(args.private_qualified_rows), args.output_root
    )
    print(canonical_json_bytes(hashes).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
